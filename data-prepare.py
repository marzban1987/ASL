import os
import shutil
import random

def split_data(source_folder, train_folder, test_folder, split_ratio=0.8):
    # ایجاد پوشه‌های Train و Test در صورت عدم وجود
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # پیمایش در پوشه‌های داخل فولدر اصلی
    for folder_name in os.listdir(source_folder):
        folder_path = os.path.join(source_folder, folder_name)
        
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            random.shuffle(files)  # فایل‌ها را به صورت تصادفی مخلوط می‌کند
            
            train_size = int(len(files) * split_ratio)
            train_files = files[:train_size]
            test_files = files[train_size:]
            
            # ایجاد پوشه‌های Train و Test برای هر فلدر
            train_folder_path = os.path.join(train_folder, folder_name)
            test_folder_path = os.path.join(test_folder, folder_name)
            
            if not os.path.exists(train_folder_path):
                os.makedirs(train_folder_path)
            if not os.path.exists(test_folder_path):
                os.makedirs(test_folder_path)
            
            # کپی کردن فایل‌ها به پوشه Train
            for file_name in train_files:
                shutil.copy(os.path.join(folder_path, file_name), os.path.join(train_folder_path, file_name))
            
            # کپی کردن فایل‌ها به پوشه Test
            for file_name in test_files:
                shutil.copy(os.path.join(folder_path, file_name), os.path.join(test_folder_path, file_name))

# مثال برای فراخوانی تابع
source_folder = 'data'
train_folder = 'Train'
test_folder = 'Test'

split_data(source_folder, train_folder, test_folder)
