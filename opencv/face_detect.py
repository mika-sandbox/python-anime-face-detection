# Face detection using OpenCV
import argparse
import cv2
import glob
import os

def main(args):
    src = args.src

    for file in glob.glob(src + '/**/*.*'):
        # 画像読み込み
        image = cv2.imread(file)
        # グレイスケール (した方が、処理速度が早いとのこと)
        grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayscaled = cv2.equalizeHist(grayscaled)
        cascade = cv2.CascadeClassifier('../lbpcascade_animeface.xml')
        facerect = cascade.detectMultiScale(grayscaled, scaleFactor=1.1, minNeighbors=3, minSize=(24, 24))
        if len(facerect) != 0:
            color = (255, 255, 255)
            for rect in facerect:
                cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), color, thickness=2)
            filepath, extension = os.path.splitext(file)
            dist = './' + os.path.basename(filepath) + '_detected' + extension
            cv2.imwrite(dist, image)
        else:
            print(file + ' ... No detected face.')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    main(parser.parse_args())

