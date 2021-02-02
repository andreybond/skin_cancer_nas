import os
import zipfile
import logging
logger = logging.getLogger('nni')



folders_to_zip = [  '/mnt/data/interim/_melanoma_20200728/loc_old_colored/D22/2018_03_12_95_4',
                    '/mnt/data/interim/_melanoma_20200728/loc_old_colored/D22/2018-11-02_186_5',
                    '/mnt/data/interim/_melanoma_20200728/loc/D22/2019-08-09_09-37-45',
                    '/mnt/data/interim/_melanoma_20200728/loc_old_colored/C43/2018_02_12_89_1',
                    '/mnt/data/interim/_melanoma_20200728/checkyourskin/C44/2019-01-30_10-54-21',
                    '/mnt/data/interim/_melanoma_20200728/loc_old_colored/C44/2018-10-26_180_4',
                    '/mnt/data/interim/_melanoma_20200728/checkyourskin/C44/2019-04-01_11-15-41',
                    '/mnt/data/interim/_melanoma_20200728/loc_old_colored/L82/2018-04-18_47B_6B',
                    '/mnt/data/interim/_melanoma_20200728/checkyourskin/L82/2019-02-07_11-45-14',
                    '/mnt/data/interim/_melanoma_20200728/loc_old_colored/L82/2018-11-09_191_6',
                    '/mnt/data/interim/_melanoma_20200728/checkyourskin/L82/2019-02-07_11-47-37',
                    '/mnt/data/interim/_melanoma_20200728/loc_old_colored/L82/2018-07-27_150_2',
                    '/mnt/data/interim/_melanoma_20200728/checkyourskin/L82/2019-03-22_12-45-25',
                    '/mnt/data/interim/_melanoma_20200728/loc_old_colored/L82/2017_10_23_8_3',
                    '/mnt/data/interim/_melanoma_20200728/loc_old_colored/L82/2017_12_04_50_2',
                    '/mnt/data/interim/_melanoma_20200728/checkyourskin/L82/2019-01-25_11-39-37',
                    '/mnt/data/interim/_melanoma_20200728/loc_old_colored/D22/2018-05-30_125_4',
                    '/mnt/data/interim/_melanoma_20200728/loc/D22/2019-12-20_12-34-33',
                    '/mnt/data/interim/_melanoma_20200728/loc/D22/2019-12-20_12-35-48',
                    '/mnt/data/interim/_melanoma_20200728/loc_old_colored/D22/2017_11_3__3_1',
                    '/mnt/data/interim/_melanoma_20200728/loc/D22/2019-12-20_12-34-58',
                    '/mnt/data/interim/_melanoma_20200728/loc/D22/2019-09-18_10-50-49',
                    '/mnt/data/interim/_melanoma_20200728/loc_old_colored/D22/2017_11_3__3_2',
                    '/mnt/data/interim/_melanoma_20200728/loc/D22/2019-12-20_12-33-09',
                    '/mnt/data/interim/_melanoma_20200728/loc/D22/2019-12-20_12-36-13',
                    '/mnt/data/interim/_melanoma_20200728/loc/C44/2019-12-06_09-17-39',
                    '/mnt/data/interim/_melanoma_20200728/loc_old_colored/C44/2017_10_23_5_1',
                    '/mnt/data/interim/_melanoma_20200728/loc/L82/2019-08-19_11-00-49']

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

if __name__ == "__main__":
    for folder in folders_to_zip:
        path, fle = os.path.split(folder)
        zipf = zipfile.ZipFile('./zipped_images/{}.zip'.format(fle), 'w', zipfile.ZIP_DEFLATED)
        zipdir(folder, zipf)
        zipf.close()
        logger.info(fle)