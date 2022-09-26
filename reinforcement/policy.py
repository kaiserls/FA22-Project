class Policy(nn.Module):
    def __init__(self, hidden_size):
        super(Policy, self).__init__()
        
        visible_squares = (VISIBLE_RADIUS * 2 + 1) ** 2
        input_size = visible_squares + 1 + 2 # Plus agent health, y, x
        
        self.inp = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, 4 + 1, bias=False) # For both action and expected value

    def forward(self, x):
        x = x.view(1, -1)
        x = F.tanh(x) # Squash inputs
        x = F.relu(self.inp(x))
        x = self.out(x)
        
        # Split last five outputs into scores and value
        scores = x[:,:4]
        value = x[:,4]
        return scores, value