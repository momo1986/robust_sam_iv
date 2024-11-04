import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gc
import segpgd
#global fig_index
#fig_index = 0
criterion = nn.CrossEntropyLoss()
class_mapping = {
    "road": 0,
    "sidewalk": 1,
    "building": 2,
    "wall": 3,
    "fence": 4,
    "pole": 5,
    "traffic light": 6,
    "traffic sign": 7,
    "vegetation": 8,
    "terrain": 9,
    "sky": 10,
    "person": 11,
    "rider": 12,
    "car": 13,
    "truck": 14,
    "bus": 15,
    "train": 16,
    "motorcycle": 17,
    "bicycle": 18
}
# 解码分割区域信息
def decode_rle(rle_str, height, width):
    # 你需要实现此函数来解码运行长度编码的字符串，并返回一个布尔类型的numpy数组，表示对象的分割区域
    # 这里使用随机数据代替
    segmentation = np.random.randint(2, size=(height, width), dtype=np.bool_)
    return segmentation
# 将分割区域填充到标签矩阵中
def fill_labels(segmentation, label_matrix, bbox, class_proposals):
    x, y, w, h = bbox
    label_height, label_width = label_matrix.shape
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    label_height = int(label_height)
    label_width = int(label_width)
    if label_height < y + h:
        h = label_height - y
    if label_width < x + w:
        w = label_widght -x 
    class_value = class_mapping.get(class_proposals, 0)  # 如
    label_matrix[y:y+h, x:x+w] = class_value
    return label_matrix

def fgsm_attack(image, epsilon, data_grad):
    # 获取数据梯度的符号
    # 将梯度信息resize为与输入图像相同的尺寸
    resized_data_grad = F.interpolate(data_grad, size=(image.size(0), image.size(1)), mode='nearest')
    resized_data_grad = resized_data_grad.permute(0, 2, 3, 1).squeeze()
    sign_data_grad = resized_data_grad.sign()
    print(sign_data_grad)
    #print(resized_data_grad.shape)
    #print(image)
    #对输入图像添加扰动
    perturbed_image = image + epsilon * sign_data_grad
    # 限制扰动后的图像像素范围在 [0,1] 内
    perturbed_image = torch.clamp(perturbed_image, 0, 255).to(torch.int)
    return perturbed_image

import matplotlib.pyplot as plt
def segformer_segmentation(image, processor, model, rank, anns):
    h, w, _ = image.shape
    #print(h)
    #print(w)
    original_input = image
    epsilon = 8
    iters = 20
    alpha = 2
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    #print(inputs)
    #print(outputs)
    #print(outputs.logits.shape)
    #print(inputs['pixel_values'].shape)a
    #print(len(anns['annotations']))
    #print(intermediate_logits.shape)
    #print(anns)
    #print(anns["annotations"].shape)
    #print(anns['annotations'])
    
    label_matrix = np.zeros((1024, 2048), dtype=np.uint8)
  
    # 解析分割信息并填充标注矩阵
    for ann in anns["annotations"]:
        #print(ann)
        height, width = ann["segmentation"]["size"]
        counts = ann["segmentation"]["counts"]
        segmentation = decode_rle(counts, height, width)
        bbox = ann["bbox"]
        class_proposals = ann["segmentation"].get("class_proposals", None) 
        label_matrix = fill_labels(segmentation, label_matrix, bbox, class_proposals)
                          
    # 将标注矩阵转换成张量
    # 计算损失
    #print(image)
    image = torch.tensor(image.astype(float), requires_grad=True).to(rank)
    model.eval() 
    inputs = processor(images=image.unsqueeze(0), return_tensors="pt").to(rank)
    #print(inputs)
    #image.requires_grad = True 
    #print(inputs)
    #inputs['pixel_values'] = torch.tensor(inputs['pixel_values'], requires_grad=True)
    with torch.enable_grad():
        for i in range(iters):
            gc.collect()
            torch.cuda.empty_cache()
            inputs['pixel_values']=inputs['pixel_values'].clone().detach().requires_grad_(True)
 
            outputs = model(**inputs)
            #outputs = model(image)
            intermediate_logits = F.interpolate(outputs.logits, size=(h, w), mode='bilinear', align_corners=True)
    
            label_tensor = torch.tensor(label_matrix, dtype=torch.long).unsqueeze(0).to(rank) # 添加批量维度
            loss = criterion(torch.softmax(intermediate_logits, dim=1), label_tensor)
        #data_grad = torch.autograd.grad(loss, inputs['pixel_values'], allow_unused=True, retain_graph=False, create_graph=False
        #                            )[0]
            loss.backward()
            data_grad = inputs['pixel_values'].grad
            resized_data_grad = F.interpolate(data_grad, size=(image.size(0), image.size(1)), mode='nearest')
            resized_data_grad = resized_data_grad.permute(0, 2, 3, 1).squeeze()
            adv_image = image + alpha * resized_data_grad.sign()
            eta = torch.clamp(adv_image - image, min=-epsilon, max=epsilon)
            image = torch.clamp(image + eta, min=0, max=255).detach_()
            #image.requires_grad = True
            image = torch.tensor(image, requires_grad=True).to(rank)
            inputs = processor(images=image.unsqueeze(0), return_tensors="pt").to(rank)

        #import ipdb;ipdb.set_trace()
        #data_grad = torch.autograd.grad(loss, image, retain_graph=False, create_graph=False)[0]
        #optimizer.zero_grad()
        #loss.backward()
 
        #data_grad = image.grad
        #print(data_grad)
        #
    # 启用梯度跟踪
    #torch.set_grad_enabled(True)
    # 启用梯度跟踪
    #torch.enable_grad()
    #data_grad = image.grad.data
    #perturbed_image = fgsm_attack(image, epsilon, data_grad)
    #inputs = processor(images=perturbed_image, return_tensors="pt").to(rank)
    #print(inputs)
    outputs = model(**inputs)
    #print(image.shape)
    #print(type(image))
    #imgplot = plt.imshow(np.transpose(inputs['pixel_values'].cpu().numpy().squeeze(0), (1, 2, 0)))
    #imgplot =plt.imshow(image.cpu().numpy().astype(np.uint8))
    #plt.show()
    #global fig_index
    #fig_index = fig_index + 1
    #fig_name = str(fig_index) + ".pdf"
    #plt.savefig(fig_name)
    logits = outputs.logits
    logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=True)
    predicted_semantic_map = logits.argmax(dim=1).squeeze(0)
    return predicted_semantic_map
