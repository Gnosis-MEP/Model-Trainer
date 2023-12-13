from PIL import Image
import cv2
import numpy as np



class CVDiffModel():
    def __init__(self):
        self.prev_key_frame = None
        self.diff_threshold = 0.0

    def predict(self, new_image_frame):
        perc_difference = 1.0
        if self.prev_key_frame is not None:
            #--- take t he absolute difference of the images ---
            abs_diff = cv2.absdiff(self.prev_key_frame, new_image_frame)

            #--- convert the result to integer type ---
            abs_diff_int = abs_diff.astype(np.uint8)

            #--- find percentage difference based on number of pixels that are not zero ---
            perc_difference = np.count_nonzero(abs_diff_int)/ abs_diff_int.size

        if perc_difference > self.diff_threshold:
            del self.prev_key_frame
            self.prev_key_frame = new_image_frame
        return perc_difference





def get_base_diff_model():
    return CVDiffModel
# https://stackoverflow.com/questions/51288756/how-do-i-calculate-the-percentage-of-difference-between-two-images-using-python


if __name__ == '__main__':
    model = get_base_diff_model()()
    model.diff_threshold = 0.5

    image1 = cv2.cvtColor(np.array(Image.open(
        '/home/arruda/projects/my-gnosis/live-street-datasets/my-creations/selected/Frames/TS-D-B-2/frame_1.png'
    )), cv2.COLOR_RGB2BGR)
    image2 = cv2.cvtColor(np.array(Image.open(
        '/home/arruda/projects/my-gnosis/live-street-datasets/my-creations/selected/Frames/TS-D-B-2/frame_2.png'
    )), cv2.COLOR_RGB2BGR)
    image3 = cv2.cvtColor(np.array(Image.open(
        '/home/arruda/projects/my-gnosis/live-street-datasets/my-creations/selected/Frames/TS-D-B-2/frame_10.png'
    )), cv2.COLOR_RGB2BGR)
    image4 = cv2.cvtColor(np.array(Image.open(
        '/home/arruda/projects/my-gnosis/live-street-datasets/my-creations/selected/Frames/TS-D-B-2/frame_91.png'
    )), cv2.COLOR_RGB2BGR)

    image5 = cv2.cvtColor(np.array(Image.open(
        '/home/arruda/projects/my-gnosis/live-street-datasets/my-creations/selected/Frames/TS-D-B-2/frame_101.png'
    )), cv2.COLOR_RGB2BGR)


    import ipdb; ipdb.set_trace()
    res1 = model.predict(image1)
    print(res1)
    res2 = model.predict(image2)
    print(res2)
    res3 = model.predict(image3)
    print(res3)
    res4 = model.predict(image4)
    print(res4)
    res5 = model.predict(image5)
    print(res5)
