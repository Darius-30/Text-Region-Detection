import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def process_and_save_mser(image_path, output_folder):
    # Citim imaginea
    img = cv2.imread(image_path)
    if img is None:
        print(f"Eroare: Nu am putut citi imaginea {os.path.basename(image_path)}")
        return None
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output_img = img_rgb.copy()

    # 1. Inițializare MSER
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)

    valid_boxes = []

    # Analizăm fiecare regiune
    for region in regions:
        x, y, w, h = cv2.boundingRect(region)
        area = w * h
        
        # Filtrare Geometrică
        if area < 80 or area > 10000: continue
        aspect_ratio = float(w) / h
        if aspect_ratio < 0.1 or aspect_ratio > 4.0: continue

        # Stroke Width / Distance Transform
        mask_char = np.zeros((h, w), dtype=np.uint8)
        local_region = region - [x, y]
        cv2.fillPoly(mask_char, [local_region], 255)
        dist_transform = cv2.distanceTransform(mask_char, cv2.DIST_L2, 3)
        mean_val, std_val = cv2.meanStdDev(dist_transform, mask=mask_char)
        
        if mean_val[0][0] > 0:
            thickness_variation = std_val[0][0] / mean_val[0][0]
            if thickness_variation > 0.8: continue
                
        valid_boxes.append((x, y, w, h))

    # GRUPAREA LITERELOR ÎN CUVINTE (Morphological Dilation)
     
    # Creăm o mască neagră de dimensiunea imaginii
    mask_words = np.zeros_like(gray)
    
    # Desenăm dreptunghiurile literelor (pline cu alb) pe mască
    for (x, y, w, h) in valid_boxes:
        cv2.rectangle(mask_words, (x, y), (x + w, y + h), 255, -1)
        
    # Definim un element structural (kernel) de formă dreptunghiulară (lung pe orizontală)
    # Valorile (15, 3) înseamnă că dilatăm mult stânga-dreapta (15 pixeli) și puțin sus-jos (3 pixeli)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    
    # Aplicăm dilatarea pentru a contopi literele apropiate
    dilated_mask = cv2.dilate(mask_words, kernel, iterations=2)
    
    # Găsim noile contururi (blocurile de cuvinte contopite)
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Desenăm cutiile finale (pe cuvinte) pe imaginea de ieșire
    for contour in contours:
        # Putem filtra și aici blocurile prea mici apărute accidental
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # SALVARE: Convertim înapoi în BGR pentru OpenCV și salvăm
    file_name = os.path.basename(image_path)
    save_path = os.path.join(output_folder, f"detectat_{file_name}")
    cv2.imwrite(save_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    
    return output_img

# CONFIGURARE ȘI RULARE

# 1. Calea către poze
dataset_path = r"C:\Users\Bogdan\Desktop\proiect TDAV\poze MSER bune" 

# 2. Calea către folderul cu rezultate
output_dir = "rezultate_mser"

# Creăm folderul dacă nu există
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Folderul '{output_dir}' a fost creat.")

# Căutăm pozele
search_path = os.path.join(dataset_path, '*.jpg')
image_files = glob.glob(search_path)[:10] 

if not image_files:
    print("Nu am găsit imagini!")
else:
    print(f"Procesez {len(image_files)} imagini și le salvez în '{output_dir}'...")
    
    for img_path in image_files:
        process_and_save_mser(img_path, output_dir)
        
    print("Finalizat! Verifică folderul 'rezultate_mser'.")