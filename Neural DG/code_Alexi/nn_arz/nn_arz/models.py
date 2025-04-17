import torch

import torch.nn as nn

from typing import Tuple

# Base ab-stencil CNN for cnn density modules
class ab_stencil_CNN(nn.Module):
    def __init__(self,
                 stencil: Tuple[int,int],
                 hidden: int = 20,
                 depth: int = 1,
                 activation = nn.ReLU,
                 in_channels=1,
                 out_channels=1,
                 dropout=0,
                 dtype: torch.dtype = torch.float64):
        """CNN with a (a,b) assymmetric kernel size"""
        super(ab_stencil_CNN, self).__init__()

        try:
            a, b = stencil
        except:
            raise ValueError("The stencil must be (a:int, b:int)")
        
        #store stencil as buffer
        self.register_buffer('stencil',torch.tensor(stencil))
        self.register_buffer(type(activation()).__name__,torch.zeros((1)))

        #add convolution layers in sequential module
        self.conv = nn.Sequential()
        self.conv.add_module( "Conv1",
            nn.Conv1d(  in_channels=in_channels,
                        out_channels=hidden,
                        kernel_size=(a+b+1),
                        padding = 0,
                        dtype=dtype)  )
        self.conv.add_module("act1", activation() )
        for i in range(depth-1):
            #the hidden FC layers
            self.conv.add_module( "fc_"+str(i),
                nn.Conv1d(  in_channels=hidden,
                        out_channels=hidden,
                        kernel_size=1,
                        padding = 0,
                        dtype=dtype)
                        )
            self.conv.add_module("relu"+str(i), activation())
            if dropout > 0:
                self.conv.add_module("dropout" + str(i), nn.Dropout(p=dropout))
        #final layers
        self.conv.add_module( "fc_end",
            nn.Conv1d(  in_channels=hidden,
                        out_channels=out_channels,
                        kernel_size=1,
                        padding = 0,
                        dtype=dtype) )
        # self.conv.add_module("end_relu", nn.ReLU() )


    
    def forward(self, x):
        #Pads the input tensor before convolution: 
        # Here we impose Free Flow condition (dF/dx = 0) at both L/R boundaries
        a,b = self.stencil
        # print(x.shape, x[:,:,[0]].repeat(1,1,1,a).shape, x[:,:,[-1]].repeat(1,1,1,b).shape)
        x = torch.cat((x[:,:,[0]].repeat(1,1,a, 1), x, x[:,:,[-1]].repeat(1,1,b, 1),) , dim=2,).to(x.device)
        # print(x.shape)
        # x = F.pad( x, pad=(self.a,self.b) )
        return self.conv(x.squeeze(1).transpose(1,2)).transpose(1,2).unsqueeze(1)

class flowModel(nn.Module):
    def __init__(self, 
                stencil: Tuple[int,int],
                depth: int = 2,
                hidden: int = 10,
                dx: float = 1.,
                dt: float = 1.,
                activation = nn.ReLU,
                in_channels=1,
                dropout=0,
                dtype=torch.float64,
                clamp=1.
                ):
        #register dx and dt at parent class level
        super(flowModel, self).__init__()

        self.dx = dx
        self.dt = dt
        self.clamp = clamp

        self.flowModel = ab_stencil_CNN(stencil, hidden, depth, activation=activation, dropout=dropout, in_channels=in_channels, dtype=dtype)

    def forward(self, x, density=False):
        #Predict right flow
        # flowR = self.flowModel(x)
        # flowR = torch.clamp(flowR, min=-1., max=2.)

        ########
        # flow = torch.clamp(self.flowModel(x), min=0., max=self.clamp)

        # flowL = torch.cat((flow, flow[:,:,0].unsqueeze(-1)) , dim=2).to(x.device)
        # flowR = torch.cat((flow[:,:,0].unsqueeze(-1), flow) , dim=2).to(x.device)
        # flowR = (flowR[:,:,1:] - flowR[:,:,:-1])
        # flowL = (flowL[:,:,1:] - flowL[:,:,:-1])

        # flow = torch.where(abs(flowL) > abs(flowR), flowR, flowL)
        # flow = torch.where(flowL*flowR < 0, torch.zeros_like(flow), flow)
        # return x - (self.dt/self.dx) * flow, flow
        ########
        flowR = torch.clamp(self.flowModel(x), min=0., max=self.clamp)
        flowR = torch.cat((flowR[:,:,0].unsqueeze(-1), flowR) , dim=2).to(x.device)
        return x - (self.dt/self.dx) * (flowR[:,:,1:] - flowR[:,:,:-1]), flowR



        # return x - (self.dt/self.dx) * flowR, flowR

class flowModel_x(nn.Module):
    def __init__(self, 
                stencil: Tuple[int,int],
                depth: int = 2,
                hidden: int = 10,
                dx: float = 1.,
                dt: float = 1.,
                activation = nn.ReLU,
                in_channels=2,
                dtype=torch.float64,
                ):
        #register dx and dt at parent class level
        super(flowModel_x, self).__init__()

        self.dx = dx
        self.dt = dt
        

        self.flowModel = ab_stencil_CNN(stencil, hidden, depth, activation=activation, in_channels=in_channels, dtype=dtype)

    def forward(self, x, pos):
        pos = torch.tanh(100*pos)
        a = torch.cat((x, pos), dim=-1)
        #Predict right flow
        flowR = torch.clamp(self.flowModel(a), min=0., max=.5)
        # flowR = torch.clamp(self.flowModel(x), min=0., max=1.)

        flowR = torch.cat((flowR[:,:,0].unsqueeze(-1), flowR) , dim=2).to(x.device)

        #add flows to cells
        # return x - (self.dt/self.dx) * (flowR[:,1:,] - flowR[:,:-1,])
        return x - (self.dt/self.dx) * (flowR[:,:,1:] - flowR[:,:,:-1]), flowR
        # return x - (self.dt/self.dx) * flowR, flowR

class speedModel(nn.Module):
    def __init__(self, 
                stencil: Tuple[int,int],
                depth: int = 2,
                hidden: int = 10,
                dx: float = 1.,
                dt: float = 1.,
                activation = nn.ReLU,
                in_channels=1,
                dtype=torch.float64,
                ):
        #register dx and dt at parent class level
        super(speedModel, self).__init__()

        self.dx = dx
        self.dt = dt
        

        self.speedModel = ab_stencil_CNN(stencil, hidden, depth, activation=activation, in_channels=in_channels, dtype=dtype)

    def forward(self, x):
        #Predict right flow
        # print("CLAMP TO 1")
        # flowR = torch.mul(x, torch.clamp(self.speedModel(x/4), min=0, max=1.))
        # flowR = torch.clamp(torch.mul(x, self.speedModel(x/4)), min=0, max=1.)
        # flowR = torch.clamp(torch.mul(x, torch.clamp(self.speedModel(x), min=0., max=1.)), min=0, max=1.)
        flowR = torch.mul(x, self.speedModel(x))
        # flowR = torch.clamp(torch.mul(x, self.speedModel(x)), min=0., max=4.)
        # BOUNDARY CONDITION
        # Define left entering flow:  we use here the free flow condition
        # i.e the input flow of the first cell is equal to the output flow. 
        # Then the first cell stays constant in density 
        # flowR =  torch.cat((flowR[:,0,:].unsqueeze(1), flowR) , dim=1).to(x.device)
        flowR = torch.cat((flowR[:,:,0].unsqueeze(-1), flowR) , dim=2).to(x.device)

        #add flows to cells
        # return x - (self.dt/self.dx) * (flowR[:,1:,] - flowR[:,:-1,])
        return x - (self.dt/self.dx) * (flowR[:,:,1:] - flowR[:,:,:-1]), flowR

class speedModelDG(nn.Module):
    def __init__(self, 
                depth: int = 2,
                hidden: int = 10,
                dx: float = 1.,
                dt: float = 1.,
                activation = nn.ReLU,
            ):
        super(speedModelDG, self).__init__()

        self.dx = dx
        self.dt = dt
        
        self.model = nn.Sequential()
        self.model.add_module("lin0", nn.Linear(2, hidden))
        self.model.add_module("act0", activation())
        for i in range(1, depth-1):
            self.model.add_module(f"lin{str(i)}", nn.Linear(hidden, hidden))
            self.model.add_module(f"act{str(i)}", activation())
        self.model.add_module("lin_end", nn.Linear(hidden, 1))


    def forward(self, x, y):
        batch_size = x.size(0)
        inpt = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), dim=2).reshape(-1, 2)#.transpose(0, 2).reshape(2, -1).transpose(0, 1)#/4

        # inpt = (inpt[:, 0]*100 + inpt[:, 1]).unsqueeze(1)
        # return inpt#.reshape(batch_size, -1)
        # print(inpt.shape, self.model(inpt).shape)
        # print(torch.clamp(self.model(inpt.reshape(-1, 2)).reshape(batch_size, -1), min=0., max=4.).shape)
        # print(torch.clamp(self.model(inpt), min=0., max=4.).shape)
        return torch.clamp(self.model(inpt), min=0., max=1.).reshape(batch_size, -1)

        return torch.clamp(self.model(inpt.reshape(-1, 2)).reshape(batch_size, -1), min=0., max=4.)
        print(inpt.shape, torch.mul(x, self.model(inpt.reshape(-1, 2)).reshape(batch_size, -1)).shape)
        return inpt
        # return torch.mul(x, self.model(inpt.reshape(-1, 2)).reshape(batch_size, -1))
        
class punn(nn.Module):
    def __init__(self):
        #register dx and dt at parent class level
        super(punn, self).__init__()

        self.linear1 = nn.Linear(2, 10)
        self.linear2 = nn.Linear(10, 10)
        self.linear3 = nn.Linear(10, 10)
        self.linear4 = nn.Linear(10, 2)

    def forward(self, x, t): # returns rho, v
        x = torch.cat((x, t), dim=1).float()
        x = self.linear1(x)
        # x = torch.relu(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        # x = torch.relu(x)
        x = torch.tanh(x)
        x = self.linear3(x)
        # x = torch.relu(x)
        x = torch.tanh(x)
        x = self.linear4(x)
        return x[:, 0], x[:, 1]

# class flowModel(nn.Module):
#     def __init__(self, dt, dx, nx):
#         super(flowModel, self).__init__()
        
#         self.dt = dt
#         self.dx = dx

#         self.flux = nn.Sequential(
#             nn.Linear(4, 16),
#             nn.ELU(),
#             nn.Linear(16, 16),
#             nn.ELU(),
#             nn.Linear(16, 16),
#             nn.ELU(),
#             nn.Linear(16, 16),
#             nn.ELU(),
#             nn.Linear(16, 16),
#             nn.ELU(),
#             nn.Linear(16, 2)
#         )
#         # With tanh
#         # self.flux = nn.Sequential(
#         #     nn.Linear(4, 16),
#         #     nn.Tanh(),
#         #     nn.Linear(16, 16),
#         #     nn.Tanh(),
#         #     nn.Linear(16, 16),
#         #     nn.Tanh(),
#         #     nn.Linear(16, 16),
#         #     nn.Tanh(),
#         #     nn.Linear(16, 16),
#         #     nn.Tanh(),
#         #     nn.Linear(16, 2)
#         # )

#          #nn.Linear(4, 2)
    
#     def forward(self, u):
#         q_i_q_i_plus_1 = u.unsqueeze(2).repeat(1, 1, 2, 1)
#         q_i_q_i_plus_1[:, :, 0] = torch.roll(q_i_q_i_plus_1[:, :, 0], 0, 1)
#         q_i_q_i_plus_1[:, :, 1] = torch.roll(q_i_q_i_plus_1[:, :, 1], -1, 1)
#         q_i_q_i_plus_1 = q_i_q_i_plus_1.view(-1, 4)

#         q_i_q_i_plus_1 = self.flux(q_i_q_i_plus_1)
#         q_i_q_i_plus_1 = q_i_q_i_plus_1.view(u.size(0), u.size(1), -1)

#         # q_i_q_i_plus_1 = q_i_q_i_plus_1[: , :, :2]
#         # print((q_i_q_i_plus_1 - torch.roll(q_i_q_i_plus_1, 1, 1)).detach().abs().mean())

#         # print(q_i_q_i_plus_1.size())
#         # u = u + self.dt/self.dx * (q_i_q_i_plus_1 - torch.roll(q_i_q_i_plus_1, 1, 1))
#         u = u - self.dt/self.dx * (q_i_q_i_plus_1 - torch.roll(q_i_q_i_plus_1, 1, 1))

#         # u = u + q_i_q_i_plus_1
#         # q_i_q_i_plus_1 = q_i_q_i_plus_1.view(q_i_q_i_plus_1.size(0), q_i_q_i_plus_1.size(1), -1)
#         # Boundary conditions
#         u[:, 0] = u[:, 2]
#         u[:, 1] = u[:, 2]
#         # u[:, 2] = u[:, 3]
#         u[:, -1] = u[:, -3]
#         u[:, -2] = u[:, -3]
#         # u[:, -3] = u[:, -4]
#         # print(u.size())
#         return u

class refineModel(nn.Module):
    def __init__(self, 
                stencil: Tuple[int,int],
                depth: int = 2,
                hidden: int = 10,
                dx: float = 1.,
                dt: float = 1.,
                activation = nn.ReLU,
                in_channels=1,
                dtype=torch.float64,
                ):
        #register dx and dt at parent class level
        super(refineModel, self).__init__()

        self.dx = dx
        self.dt = dt
        

        self.flowModel = ab_stencil_CNN(stencil, hidden, depth, activation=activation, in_channels=in_channels, dtype=dtype)

    def forward(self, x):
        return torch.clamp(self.flowModel(x), min=0., max=4.)


class arzModel(nn.Module):
    def __init__(self, 
                stencil: Tuple[int,int],
                depth: int = 2,
                hidden: int = 10,
                dx: float = 1.,
                dt: float = 1.,
                activation = nn.ReLU,
                in_channels=2,
                out_channels=1,
                dtype=torch.float64,
                ):
        #register dx and dt at parent class level
        super(arzModel, self).__init__()

        self.dx = dx
        self.dt = dt
        

        self.arzModel = ab_stencil_CNN(stencil, hidden, depth, activation=activation, in_channels=in_channels, out_channels=out_channels, dtype=dtype)

    def forward(self, x, v, return_flow=False):
        a = torch.cat((x, v), dim=-1)
        # return
        #Predict right flow
        flowR = torch.clamp(self.arzModel(a), min=0., max=2.)
        # flowR = torch.clamp(self.flowModel(x), min=0., max=1.)

        flowR = torch.cat((flowR[:,:,0].unsqueeze(-1), flowR) , dim=2).to(x.device)

        if not return_flow:
            return x - (self.dt/self.dx) * (flowR[:,:, 1:] - flowR[:, :, :-1]) #, flowR
        else:
            return x - (self.dt/self.dx) * (flowR[:,:, 1:] - flowR[:, :, :-1]), flowR
        # return x - (self.dt/self.dx) * flowR, flowR

        # x_save = x.clone()
        # for i in range(2):
        #     a = torch.cat((x, v), dim=-1)
        #     # return
        #     #Predict right flow
        #     flowR = torch.clamp(self.arzModel(a), min=0., max=2.)
        #     # flowR = torch.clamp(self.flowModel(x), min=0., max=1.)

        #     flowR = torch.cat((flowR[:,:,0].unsqueeze(-1), flowR) , dim=2).to(x.device)

        #     x = x_save - (self.dt/self.dx) * (flowR[:,:, 1:] - flowR[:, :, :-1]) * (1/2 if i == 0 else 1.)#, flowR
        #     # return x - (self.dt/self.dx) * flowR, flowR

        # return x
