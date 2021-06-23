from pycocotools.coco import COCO
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import requests
import threading
import os
import shutil

class bcolors:
    HEADER = '\033[95m'

    INFO = '    [INFO] | '
    OKBLUE = '\033[94m[DOWNLOAD] | '
    WARNING = '\033[93m    [WARN] | '
    FAIL = '\033[91m   [ERROR] | '

    OKGREEN = '\033[92m'
    ENDC = '\033[0m'


# Truncates numbers to N decimals
def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


def convert_anns(coco, image, catIds):
    im = image
    dw = 1. / im['width']
    dh = 1. / im['height']

    annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    filename = im['file_name'].replace(".jpg", ".txt")
    print(filename)

    with open("labels/" + filename, "a") as myfile:
        for i in range(len(anns)):
            xmin = anns[i]["bbox"][0]
            ymin = anns[i]["bbox"][1]
            xmax = anns[i]["bbox"][2] + anns[i]["bbox"][0]
            ymax = anns[i]["bbox"][3] + anns[i]["bbox"][1]

            x = (xmin + xmax)/2
            y = (ymin + ymax)/2

            w = xmax - xmin
            h = ymax-ymin

            x = x * dw
            w = w * dw
            y = y * dh
            h = h * dh

            # Note: This assumes a single-category dataset, and thus the "0" at the beginning of each line.
            if (anns[i]["category_id"] == 37):
                clsid = '1 '
            else:
                clsid = '0 '
            mystring = str(clsid + str(truncate(x, 7)) + " " + str(truncate(y, 7)
                                                                   ) + " " + str(truncate(w, 7)) + " " + str(truncate(h, 7)))
            myfile.write(mystring)
            myfile.write("\n")

    myfile.close()
    return


def main():
    bc = bcolors
    folder = 'downloaded_images'
    lfolder = 'labels'

    if (os.path.exists('./' + folder)):
        shutil.rmtree('./' + folder)

    if (os.path.exists('./' + lfolder)):
        shutil.rmtree('./' + lfolder)
    
    os.mkdir('./' + folder)
    os.mkdir('./' + lfolder)

    # Download instances_train2017.json from the COCO website and put in the same directory as this script
    coco = COCO('instances_train2017.json')
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))


    # Replace category with whatever is of interest to you
    cat = ['person', 'sports ball']
    catIds = coco.getCatIds(catNms=cat)
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)

    # Create a subfolder in this directory called "downloaded_images". This is where your images will be downloaded into.
    # Comment this entire section out if you don't want to download the images
    threads = 20
    pool = ThreadPool(threads)
    if len(images) > 0:
        print(bc.INFO + 'Download of {} images in {}.'.format(len(images), folder) + bc.ENDC)
        commands = []
        for image in images:
            url = image['coco_url']
            command = 'wget -P ./downloaded_images -q ' + url
            commands.append(command)

        list(tqdm(pool.imap(os.system, commands), total=len(commands)))

        print(bc.INFO + 'Done!' + bc.ENDC)
        pool.close()
        pool.join()
    else:
        print(bc.INFO + 'All images already downloaded.' + bc.ENDC)


    # Create a subfolder in this directory called "labels". This is where the annotations will be saved in YOLO format
    threads = []
    for image in images:
        thread = threading.Thread(target=convert_anns, args=(coco, image, catIds))
        thread.daemon = True
        threads.append(thread)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return


if __name__ == '__main__':
    main()
