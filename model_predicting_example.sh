
#python predict.py flowers/test/12/image_04023.jpg checkpoint.pth --category_name cat_to_name.json --top_k 5 --gpu
#python predict.py flowers/test/30/image_03482.jpg checkpoint.pth --category_name cat_to_name.json --top_k 5 --gpu
#python predict.py flowers/train/5/image_05153.jpg checkpoint.pth --category_name cat_to_name.json --top_k 5 --gpu
python predict.py uploaded_images/french_marigold.jpg checkpoint.pth --category_name cat_to_name.json --top_k 5 --gpu
