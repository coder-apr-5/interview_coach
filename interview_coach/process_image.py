from PIL import Image
import sys

def remove_white_background(image_path, output_path):
    img = Image.open(image_path)
    img = img.convert("RGBA")
    
    datas = img.getdata()
    
    new_data = []
    # Loop through each pixel
    for item in datas:
        # If the pixel is white (with a small tolerance)
        if item[0] > 240 and item[1] > 240 and item[2] > 240:
            new_data.append((255, 255, 255, 0)) # Make it transparent
        else:
            new_data.append(item)
            
    img.putdata(new_data)
    img.save(output_path, "PNG")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        remove_white_background(sys.argv[1], sys.argv[2])
