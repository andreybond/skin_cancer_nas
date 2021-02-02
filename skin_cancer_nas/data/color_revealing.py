import os
import cv2      # pip install opencv-python

class ColorRevealer:
    def __init__(self, filepath: str, convert_now=True, evaluate_now=True):
        self.filepath = filepath
        self.orig_bgr = self.__load_original_image()
        self.hsv_img = None
        self.max_saturation = -1
        self.mean_hue = -1
        self.norm_h = None     # normalized H channel. MatLab = hsv(:,:,1) , Python = hsv[:,:,0]    Hue image.
        self.norm_s = None     # normalized S channel. MatLab = hsv(:,:,2) , Python = hsv[:,:,1]    Saturation image.
        self.norm_v = None     # normalized V channel. MatLab = hsv(:,:,3) , Python = hsv[:,:,2]    Value(intensity) image.
        self.color = ''

        if convert_now:
            self.to_hsv()
        if evaluate_now:
            self.evaluate_color()

    def __load_original_image(self):
        return cv2.imread(self.filepath)

    def to_hsv(self):
        self.hsv_img = cv2.cvtColor(self.orig_bgr, cv2.COLOR_BGR2HSV)

    def evaluate_color(self):
        self.norm_h = self.hsv_img[:, :, 0] / 255
        self.norm_s = self.hsv_img[:, :, 1] / 255
        self.norm_v = self.hsv_img[:, :, 2] / 255
        self.max_saturation = self.norm_s.max()
        if self.max_saturation < 0.5:
            self.color = 'IR'
        else:
            # # self.mean_hue = mean(mean(hsv(hsv(:,:, 3) > 0.5))); % calculated only for lighter pixels, coz only skin matters
            mask = self.norm_v > 0.5
            valid_hues = self.norm_h[mask]
            self.mean_hue = valid_hues.mean()
            # original rules
            # if 0.25 < self.mean_hue < 0.5:      # if (mean_hue > 0.25) & & (mean_Hue < 0.5) % if Hue is green tone
            #     self.color = 'G'
            # elif self.mean_hue > 0.83 or self.mean_hue < 0.17:  # elseif(mean_Hue > 0.83) | | (mean_Hue < 0.17) % if Hue is red tone
            #     self.color = 'R'

            # our empirically obtained:
            if 0.0 <= self.mean_hue < 0.001:
                self.color = 'R'
            elif 0.001 <= self.mean_hue < 0.1:
                self.color = 'UV'
            elif 0.2 <= self.mean_hue < 0.3:
                self.color = 'G'
            elif 0.5 <= self.mean_hue < 0.6:
                self.color = 'IR'
            else:
                self.color = 'HZ'

    def get_color(self):
        if not self.color:
            self.evaluate_color()
        return self.color


if __name__ == '__main__':

    mypath = r'D:\YC\1\1'
    #onlyfiles = [f for f in os.listdir(mypath) if (os.path.isfile(os.path.join(mypath, f) and f[-4:]=='.png'))]
    onlyfiles = [f for f in os.listdir(mypath) if f[-4:] == '.png']
    for filename in onlyfiles:
        cr = ColorRevealer(filepath=os.path.join(mypath, filename))
        print(filename + ' : ' + cr.color + ' (' + 'max_saturation=' + str(cr.max_saturation)+'; mean_hue='+str(cr.mean_hue)+')')


    # Playground for experiments:


    # cr = ColorRevealer(filepath=r'D:\YC\3\3\a_2017-10-18_11-12-32_38.png')
    # #cr = ColorRevealer(filepath=r'D:\YC\bacteria_data\bact_prediction_all_410_2019.11.11_15-00_model_grayscale_16-00-05.png')
    #
    # print('max_saturation: ' + str(cr.max_saturation))
    # print('mean_hue: ' + str(cr.mean_hue))
    # print('color: ' + cr.color)
    #
    # cv2.imshow('image', cr.hsv_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(cr)
