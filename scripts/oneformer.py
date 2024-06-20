import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
criterion = nn.CrossEntropyLoss()
import gc
import os
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
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
    #print(image)
    resized_data_grad = F.interpolate(data_grad, size=(image.shape[0], image.shape[1]), mode='nearest')
    resized_data_grad = resized_data_grad.permute(0, 2, 3, 1).squeeze()
    if isinstance(image, np.ndarray):
        image = torch.tensor(image)
    if isinstance(data_grad, np.ndarray):
        data_grad = torch.tensor(data_grad)
    #print(image)
    sign_data_grad = resized_data_grad.sign()
    #print(sign_data_grad)
    #print(epsilon)
    #print(sign_data_grad)
    #print(resized_data_grad.shape)
    #print(image)
    #对输入图像添加扰动
    #print(image.device)
    #print(sign_data_grad.device)
    perturbed_image = image + epsilon * sign_data_grad.cpu()
    # 限制扰动后的图像像素范围在 [0,1] 内
    perturbed_image = torch.clamp(perturbed_image, 0, 255).to(torch.int)
    #print(perturbed_image)
    return perturbed_image


def oneformer_coco_segmentation(image, oneformer_coco_processor, oneformer_coco_model, rank):
    inputs = oneformer_coco_processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(rank)
    outputs = oneformer_coco_model(**inputs)
    predicted_semantic_map = oneformer_coco_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]])[0]
    return predicted_semantic_map

def oneformer_ade20k_segmentation(image, oneformer_ade20k_processor, oneformer_ade20k_model, rank):
    inputs = oneformer_ade20k_processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(rank)
    outputs = oneformer_ade20k_model(**inputs)
    predicted_semantic_map = oneformer_ade20k_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]])[0]
    return predicted_semantic_map

def oneformer_cityscapes_segmentation(image, oneformer_cityscapes_processor, oneformer_cityscapes_model, rank, anns):
    image_array = np.array(image)
    #print(image_array.shape)
    h, w, _ = image_array.shape
    original_input = image_array
    epsilon = 16.0
    #print(image_array)
    optimizer = optim.SGD(oneformer_cityscapes_model.parameters(), lr=0.001, momentum=0.9)
    '''
    label_matrix = np.zeros((1024, 2048), dtype=np.uint8)
  
    # 解析分割信息并填充标注矩阵
    #print(anns)
    
    for ann in anns["annotations"]:
        print(ann)
        height, width = ann["segmentation"]["size"]
        counts = ann["segmentation"]["counts"]
        segmentation = decode_rle(counts, height, width)
        bbox = ann["bbox"]
        class_proposals = ann["segmentation"].get("class_proposals", None) 
        label_matrix = fill_labels(segmentation, label_matrix, bbox, class_proposals)
    '''                     
    # 将标注矩阵转换成张量
    # 计算损失
    #print(image)
    #image = torch.tensor(image.astype(float), requires_grad=True).to(rank)
    oneformer_cityscapes_model.eval() 
 
    inputs = oneformer_cityscapes_processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(rank)
    #print(inputs)
    inputs['pixel_values']=inputs['pixel_values'].clone().detach().requires_grad_(True)
    with torch.enable_grad():
        gc.collect()
        torch.cuda.empty_cache()
        with autocast():
            outputs =  oneformer_cityscapes_model(**inputs)
            #print(label_matrix)
            #print(label_matrix.shape)
            #print(outputs)
            #print(outputs.class_queries_logits)
            #print(outputs.class_queries_logits.shape)
            #print(outputs.masks_queries_logits)
            #print(outputs.masks_queries_logits.shape)
            #outputs = model(image)
            label_logits = F.interpolate(outputs.masks_queries_logits.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=True)
            intermediate_logits = F.interpolate(outputs.transformer_decoder_mask_predictions.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=True)
            #label_tensor = torch.tensor(label_matrix, dtype=torch.long).unsqueeze(0).to(rank) # 添加批量维度
            #print(label_tensor.shape)
            #print(intermediate_logits.shape)
            loss = criterion(intermediate_logits, label_logits)
        #data_grad = torch.autograd.grad(loss, inputs['pixel_values'], allow_unused=True, retain_graph=False, create_graph=False
        #                            )[0]
        scaler.scale(loss).backward()
        del outputs
        del intermediate_logits
        del loss
        gc.collect()
        torch.cuda.empty_cache()
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
    data_grad = inputs['pixel_values'].grad
    #print(data_grad)
    perturbed_image = fgsm_attack(image_array, epsilon, data_grad)
    inputs = oneformer_cityscapes_processor(images=perturbed_image, task_inputs=["semantic"], return_tensors="pt").to(rank)
    outputs = oneformer_cityscapes_model(**inputs)
    predicted_semantic_map = oneformer_cityscapes_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]])[0]
    return predicted_semantic_map
