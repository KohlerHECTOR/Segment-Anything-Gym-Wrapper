from marionet import Model, Dictionary, AnimationDataset
import torch as th
import ttools


###MODEL LOADING#####
learned_dict = Dictionary(num_classes=150,
                          patch_size= (128 // 8*2,
                                      128 // 8*2),
                          num_chans=4,
                          bottleneck_size=128)


data = AnimationDataset("pong_dataset", 128)
dataloader = th.utils.data.DataLoader(data, batch_size=4,
                                          pin_memory=True, shuffle=True,
                                          drop_last=True)


model = Model(learned_dict, layer_size=8, num_layers=2, patch_size=1,
             canvas_size=128, dim_z=128, bg_color=data.bg)



model.eval()

model_checkpointer = ttools.Checkpointer("models", model)
model_checkpointer.load_latest()

with th.no_grad():
    for x, y in enumerate(dataloader):
        model(y["im"] ,None, hard=True)
