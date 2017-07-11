# Face detection using dlib
import argparse
import cv2
import dlib
import os

detector = dlib.simple_object_detector('./third_party/detector.svm')

def main(args):
    src = args.src
    image = cv2.imread(src)
    rects = detector(image)
    print(str(detector(image)))
    if len(rects) > 0:
        print('Detected ' + str(len(rects)) + ' faces')
        for rect in rects:
            cv2.rectangle(image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 255, 255), thickness=2)
        filepath, extension = os.path.splitext(src)
        dist = filepath + '_detected' + extension
        cv2.imwrite(dist, image)
    else:
        print('No detect')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    main(parser.parse_args())

