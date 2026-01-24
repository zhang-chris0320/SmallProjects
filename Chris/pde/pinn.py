import torch
import torch.nn as nn
import torch.autograd as autograd


class MLP(nn.Module):
    def __init__(self, hid_layers, hid_units, in_dim, out_dim,activation):
        super(MLP, self).__init__()
      
        self.Linears = nn.ModuleList(
            [nn.Linear(in_dim, hid_units)] +  
            [nn.Linear(hid_units, hid_units) for _ in range(hid_layers - 1)] +  
            [nn.Linear(hid_units, out_dim)] ) 
        self.activation = nn.Tanh() 
    
    def forward(self, x, t):
       
        inp = torch.cat([x, t], dim=1)
        
      
        for i, linear in enumerate(self.Linears):
            inp = linear(inp)
            if i < len(self.Linears) - 1:
                inp = self.activation(inp)
        
        return inp


def train_pinn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    model = MLP(hid_layers=4, hid_units=64, in_dim=2, out_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
 
    Nf = 10000
    x_f = (2 * torch.pi) * torch.rand(Nf, 1)
    t_f = torch.rand(Nf, 1)
    x_f, t_f = x_f.to(device), t_f.to(device)
    
   
    Nb = 2000
    t_b = torch.rand(Nb, 1).to(device)
    x_b0 = torch.zeros(Nb, 1).to(device)
    x_b1 = (2 * torch.pi) * torch.ones(Nb, 1).to(device)
    
    Ni = 2000
    x_i = (2 * torch.pi) * torch.rand(Ni, 1).to(device)
    t_i = torch.zeros(Ni, 1).to(device)
    u_i = 1 - torch.cos(x_i)  #
  
    for epoch in range(20000):
        optimizer.zero_grad()
        
       
        x_f.requires_grad = True
        t_f.requires_grad = True
        u_f = model(x_f, t_f)
       
        u_t = autograd.grad(u_f, t_f, grad_outputs=torch.ones_like(u_f),
                           create_graph=True)[0]
        u_x = autograd.grad(u_f, x_f, grad_outputs=torch.ones_like(u_f),
                           create_graph=True)[0]
        u_xx = autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_f),
                            create_graph=True)[0]
        
   
        f_res = u_t - u_xx
        loss_f = torch.mean(f_res**2)
        
       
        loss_b = torch.mean(model(x_b0, t_b)**2) + \
        torch.mean(model(x_b1, t_b)**2)
        
     
        u_pred_i = model(x_i, t_i)
        loss_i = torch.mean((u_pred_i - u_i)**2)
        
 
        loss = loss_f + loss_b + loss_i
        
     
        loss.backward()
        optimizer.step()
        
     
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    return model


def test_model(model, device):

    
    x_test = torch.linspace(0, 2*torch.pi, 100).reshape(-1, 1).to(device)
    t_test = torch.linspace(0, 1, 50).reshape(-1, 1).to(device)

    x_i_test = torch.linspace(0, 2*torch.pi, 100).reshape(-1, 1).to(device)
    t_i_test = torch.zeros(100, 1).to(device)
    
    with torch.no_grad():

        u_pred = model(x_i_test, t_i_test)
        u_true = 1 - torch.cos(x_i_test)
        
        mse_i = torch.mean((u_pred - u_true)**2).item()
        print(f"Initial condition MSE: {mse_i:.6f}")
        
       
        x_b_test = torch.tensor([[0.0], [2*torch.pi]]).to(device)
        t_b_test = torch.rand(2, 1).to(device)
        u_b_pred = model(x_b_test, t_b_test)
        print(f"Boundary values at x=0 and x=2π: {u_b_pred.cpu().numpy().flatten()}")
    
    return mse_i


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = train_pinn()
    
   
    torch.save(model.state_dict(), "pinn_heat_modulelist.pt")
    print("Model saved to pinn_heat_modulelist.pt")
    
    
    test_model(model, device)