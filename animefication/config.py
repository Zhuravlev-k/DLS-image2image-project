import torch

ROOT_DIR = './'
IMG_SIZE = 256
BATCH_SIZE = 1
STATS = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-5 
LAMBDA_IDENTITY = 0.35
LAMBDA_CYCLE = 8
NUM_WORKERS = 8
NUM_EPOCHS = 32
LOAD_MODEL = True
SAVE_MODEL = True
WEIGHTS_GEN_A = "genA_weights.tar"
WEIGHTS_GEN_B = "genB_weights.tar"
WEIGHTS_DISC_A = "discA_weights.tar"
WEIGHTS_DISC_B = "discB_weights.tar"

def initialize_conv_weights_normal(model):
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(model.weight.data, 0.0, 0.02)
        if hasattr(model, "bias") and model.bias is not None:
            torch.nn.init.constant_(model.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(model.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(model.bias.data, 0.0)

def save_weights(model, optimizer, filename):
    print(f"SAVING: {filename}")
    weights = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(weights, filename)

def load_weights(weights_file, model, optimizer, lr):
    print(f"LOADING {weights_file}")
    weights = torch.load(weights_file, map_location=DEVICE)
    model.load_state_dict(weights["state_dict"])
    optimizer.load_state_dict(weights["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def load_weights_inference(weights_file, model):
    weights = torch.load(weights_file, map_location=DEVICE)
    model.load_state_dict(weights["state_dict"])