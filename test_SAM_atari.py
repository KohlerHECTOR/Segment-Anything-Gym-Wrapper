from segment_anything import build_sam, SamAutomaticMaskGenerator, sam_model_registry
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
from timer import Timer


dir = "results_masks/"
img_name = "screenshot_pong_42000.png"
image = cv2.imread("screenshots_atari/"+img_name, 1) #bgr
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #rgb

plt.figure(figsize=(20,20))
plt.imshow(image)
plt.savefig(dir+img_name)

#vit_l, default, vit_b
#sam = sam_model_registry["vit_b"](checkpoint="superlight_model.pth") #76 sec per inf
sam = sam_model_registry["default"](checkpoint="models/full_model.pth") #100 sec per inf
# sam = sam_model_registry["vit_l"](checkpoint="light_model.pth") #90 sec per inf # BBAAAAAAAAD

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

print(len(masks))
print(masks[0].keys())
for i, mask in enumerate(masks):
    with open(dir + "mask" + str(i) + ".txt", 'w') as f:
        print(mask, file=f)

plt.clf()
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.savefig(dir +"mask_" + img_name)
