from DenseUNET import Tiramisu
from DataLoaders import transform

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = Tiramisu(3, 6)

model.to(device)
model.load_state_dict(torch.load('Tiramisu_final_weights.pth'))
model.eval()


def predict(image):
    image = transform(image).to(device)
    image = image.unsqueeze(0)
    output =  model(image)
    return    output.to('cpu').numpy()
