import shutil
from glob import glob
import random
from pprint import pprint


image_list = [
    'training_new_IMG_3451',
    'training_new_IMG_3609',
    'training_new_IMG_3611',
    'training_new_IMG_3616',
    'training_new_IMG_3620',
    'training_new_IMG_3636',
    'training_new_IMG_3639',
    'training_new_IMG_3646',
    'training_new_IMG_3674',
    'training_new_IMG_3687',
    'training_new_IMG_3689',
    'training_new_IMG_3754',
    'training_new_IMG_4099',
    'training_new_IMG_4169',
    'training_new_IMG_4220',
    'training_new_IMG_4567',
    'training_new_IMG_4568',
    'training_new_IMG_4577',
    'training_new_IMG_4934',
    'training_new_IMG_5048',
    'training_new_IMG_5227',
    'training_new_IMG_3691',
    'training_new_IMG_3661',
    'training_new_IMG_3731',
    'training_new_IMG_3738',
    'training_new_IMG_4196',
    'training_new_IMG_4267',
    'training_new_IMG_4279',
    'training_new_IMG_4330',
    'training_new_IMG_4456',
    'training_new_IMG_4487',
    'training_new_IMG_4497',
    'training_new_IMG_4502',
    'training_new_IMG_4519',
    'training_new_IMG_4538',
    'training_new_IMG_4565',
    'training_new_IMG_4720',
    'training_new_IMG_4760',
    'training_new_IMG_4778',
    'training_new_IMG_4810',
    'training_new_IMG_4822',
    'training_new_IMG_4829',
    'training_new_IMG_4839',
    'training_new_IMG_5003',
    'training_new_IMG_5004',
    'training_new_IMG_5033',
    'training_new_IMG_5046',
    'training_new_IMG_5105',
    'training_new_IMG_5196',
    'training_new_IMG_5250',
    'training_new_IMG_5253',
    'training_new_IMG_5268',
    'training_new_IMG_5291',
    'training_new_IMG_5312',
    'training_new_IMG_5320',
    'training_new_IMG_4331',
    'training_new_IMG_4524',
    'training_new_IMG_4563',
    'training_new_IMG_4668',
    'training_new_IMG_4763',
    'training_new_IMG_4816',
    'training_new_IMG_4825',
    'training_new_IMG_4993',
    'training_new_IMG_5052',
    'training_new_IMG_5058',
    'training_new_IMG_5175',
    'training_new_IMG_5286',
    'training_new_IMG_3746',
    'training_new_IMG_4460',
    'training_new_IMG_4525',
    'training_new_IMG_4562',
    'training_new_IMG_4574',
    'training_new_IMG_4781',
    'training_new_IMG_4784',
    'training_new_IMG_4811',
    'training_new_IMG_4824',
    'training_new_IMG_4931',
    'training_new_IMG_4935',
    'training_new_IMG_4989',
    'training_new_IMG_5050',
    'training_new_IMG_5065',
    'training_new_IMG_5210',
    'training_new_IMG_5244',
    'training_new_IMG_5287',
    'training_new_IMG_3411',
    'training_new_IMG_3417',
    'training_new_IMG_3444',
    'training_new_IMG_3447',
    'training_new_IMG_3482',
    'training_new_IMG_3647',
    'training_new_IMG_3885',
    'training_new_IMG_3904',
    'training_new_IMG_4419',
    'training_new_IMG_4617',
    'training_new_IMG_4702',
    'training_new_IMG_4900',
    'training_new_IMG_5123',
    'training_new_IMG_3499',
    'training_new_IMG_3525',
    'training_new_IMG_3555',
    'training_new_IMG_3593',
    'training_new_IMG_3626',
    'training_new_IMG_3845',
    'training_new_IMG_4613',
    'training_new_IMG_4908',
    'training_new_IMG_4983',
    'training_new_IMG_3456',
    'training_new_IMG_3569',
    'training_new_IMG_3581',
    'training_new_IMG_3700',
    'training_new_IMG_3743',
    'training_new_IMG_3813',
    'training_new_IMG_3833',
    'training_new_IMG_4043',
    'training_new_IMG_4246',
    'training_new_IMG_4533',
    'training_new_IMG_4551',
    'training_new_IMG_4636',
    'training_new_IMG_5028',
    'training_new_IMG_5194',
    'training_new_IMG_5205',
    'training_new_IMG_3460',
    'training_new_IMG_3595',
    'training_new_IMG_3596',
    'training_new_IMG_3794',
    'training_new_IMG_3812',
    'training_new_IMG_3862',
    'training_new_IMG_3990',
    'training_new_IMG_4002',
    'training_new_IMG_4102',
    'training_new_IMG_4232',
    'training_new_IMG_3414',
    'training_new_IMG_4087',
    'training_new_IMG_4240',
    'training_new_IMG_4486',
    'training_new_IMG_4535',
    'training_new_IMG_4594',
    'training_new_IMG_4602',
    'training_new_IMG_4870',
    'training_new_IMG_5041',
    'training_new_IMG_5170']


def select_random_photo():
    directory = "G:\\Users\\nc17\\Research\\Guatemala\\photos\\2015\\PNLT\\output2\\"
    long_list = glob(directory + "*.tif")
    return random.choice(long_list)


def select_x_random_photos(number_of_photos):
    photo_list = []
    for i in range(number_of_photos):
        photo_list.append(select_random_photo())

    return photo_list


def select_x_random_photos2(number_of_photos):
    directory = "G:\\Users\\nc17\\Research\\Guatemala\\photos\\2015\\PNLT\\output2\\"
    long_list = glob(directory + "*.tif")
    return random.sample(long_list, number_of_photos)


def main():
    # direct = "../training_data/"
    # for i in image_list:
    #     shutil.copy(direct + i + ".tif", "E:\\Research Project\\training\\")

    # directory = "G:\\Users\\nc17\\Research\\Guatemala\\photos\\2015\\PNLT\\"
    # for i in image_list:
    #     nd = i.replace("training_new_", "")
    #     # print(nd)
    #     shutil.copy(directory + nd + ".jpg", "E:\\Research Project\\training\\originals\\")

    random_photos = select_x_random_photos2(50)
    pprint(random_photos)
    for i in random_photos:
        shutil.copy(i, "E:\\Research Project\\random\\")
        trimmed1 = i.replace("output2\\new_", "")
        trimmed2 = trimmed1.replace("tif", "jpg")
        shutil.copy(trimmed2, "E:\\Research Project\\random\\originals\\")


if __name__ == "__main__":
    main()
