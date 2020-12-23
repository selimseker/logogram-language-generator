import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import random

class Omniglot(Dataset):
    def __init__(self, split="train", transform=None):
        self.split = split
        if(transform==None):
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

        if(split=="train"):
            self.alphabet_names = ['Alphabet_of_the_Magi', 'Anglo-Saxon_Futhorc', 'Arcadian', 'Armenian', 'Asomtavruli_(Georgian)', 'Balinese', 'Bengali', 'Blackfoot_(Canadian_Aboriginal_Syllabics)', 'Braille', 'Burmese_(Myanmar)', 'Cyrillic', 'Early_Aramaic', 'Futurama', 'Grantha', 'Greek', 'Gujarati', 'Hebrew', 'Inuktitut_(Canadian_Aboriginal_Syllabics)', 'Japanese_(hiragana)', 'Japanese_(katakana)', 'Korean', 'Latin', 'Malay_(Jawi_-_Arabic)', 'Mkhedruli_(Georgian)', 'N_Ko', 'Ojibwe_(Canadian_Aboriginal_Syllabics)', 'Sanskrit', 'Syriac_(Estrangelo)', 'Tagalog', 'Tifinagh']
            self.character_nums = [20, 29, 26, 41, 40, 24, 46, 14, 26, 34, 33, 22, 26, 43, 24, 48, 22, 16, 52, 47, 40, 26, 40, 41, 33, 14, 42, 23, 17, 55]

        elif(split=="test"):
            self.alphabet_names = ['Angelic', 'Atemayar_Qelisayer', 'Atlantean', 'Aurek-Besh', 'Avesta', 'Ge_ez', 'Glagolitic', 'Gurmukhi', 'Kannada', 'Keble', 'Malayalam', 'Manipuri', 'Mongolian', 'Old_Church_Slavonic_(Cyrillic)', 'Oriya', 'Sylheti', 'Syriac_(Serto)', 'Tengwar', 'Tibetan', 'ULOG']
            self.character_nums = [20, 26, 26, 26, 26, 26, 45, 45, 41, 26, 47, 40, 30, 45, 46, 28, 23, 25, 42, 26]

        else: # all splits
            self.alphabet_names = ['Alphabet_of_the_Magi', 'Anglo-Saxon_Futhorc', 'Arcadian', 'Armenian', 'Asomtavruli_(Georgian)', 'Balinese', 'Bengali', 'Blackfoot_(Canadian_Aboriginal_Syllabics)', 'Braille', 'Burmese_(Myanmar)', 'Cyrillic', 'Early_Aramaic', 'Futurama', 'Grantha', 'Greek', 'Gujarati', 'Hebrew', 'Inuktitut_(Canadian_Aboriginal_Syllabics)', 'Japanese_(hiragana)', 'Japanese_(katakana)', 'Korean', 'Latin', 'Malay_(Jawi_-_Arabic)', 'Mkhedruli_(Georgian)', 'N_Ko', 'Ojibwe_(Canadian_Aboriginal_Syllabics)', 'Sanskrit', 'Syriac_(Estrangelo)', 'Tagalog', 'Tifinagh', 'Angelic', 'Atemayar_Qelisayer', 'Atlantean', 'Aurek-Besh', 'Avesta', 'Ge_ez', 'Glagolitic', 'Gurmukhi', 'Kannada', 'Keble', 'Malayalam', 'Manipuri', 'Mongolian', 'Old_Church_Slavonic_(Cyrillic)', 'Oriya', 'Sylheti', 'Syriac_(Serto)', 'Tengwar', 'Tibetan', 'ULOG']
            self.character_nums = [20, 29, 26, 41, 40, 24, 46, 14, 26, 34, 33, 22, 26, 43, 24, 48, 22, 16, 52, 47, 40, 26, 40, 41, 33, 14, 42, 23, 17, 55, 20, 26, 26, 26, 26, 26, 45, 45, 41, 26, 47, 40, 30, 45, 46, 28, 23, 25, 42, 26]

        self.n_images = sum(self.character_nums) * 20

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        j = idx % 20
        i = idx // 20
        label = i

        for k in range(len(self.alphabet_names)):
            if(i < self.character_nums[k]):
                alphabet = self.alphabet_names[k]+"/"
                if(self.split=="train"):
                    folder = "images_background/"
                elif(self.split=="test"):
                    folder = "images_evaluation/"
                else:
                    folder = "images_background/" if k<30 else "images_evaluation/"
                break
            else:
                i -= self.character_nums[k]

        char_name = "omniglot/"+folder+alphabet+"character"+(str(i+1) if i>8 else "0"+str(i+1))+"/"
        example_list = sorted(os.listdir(char_name))
        img = Image.open(char_name+example_list[j])

        return (self.transform(img.convert('L')), label)

class Omniglot_triplet(Dataset):
    def __init__(self, split="train", transform=None):
        self.split = split
        if(transform==None):
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

        if(split=="train"):
            self.alphabet_names = ['Alphabet_of_the_Magi', 'Anglo-Saxon_Futhorc', 'Arcadian', 'Armenian', 'Asomtavruli_(Georgian)', 'Balinese', 'Bengali', 'Blackfoot_(Canadian_Aboriginal_Syllabics)', 'Braille', 'Burmese_(Myanmar)', 'Cyrillic', 'Early_Aramaic', 'Futurama', 'Grantha', 'Greek', 'Gujarati', 'Hebrew', 'Inuktitut_(Canadian_Aboriginal_Syllabics)', 'Japanese_(hiragana)', 'Japanese_(katakana)', 'Korean', 'Latin', 'Malay_(Jawi_-_Arabic)', 'Mkhedruli_(Georgian)', 'N_Ko', 'Ojibwe_(Canadian_Aboriginal_Syllabics)', 'Sanskrit', 'Syriac_(Estrangelo)', 'Tagalog', 'Tifinagh']
            self.character_nums = [20, 29, 26, 41, 40, 24, 46, 14, 26, 34, 33, 22, 26, 43, 24, 48, 22, 16, 52, 47, 40, 26, 40, 41, 33, 14, 42, 23, 17, 55]

        elif(split=="test"):
            self.alphabet_names = ['Angelic', 'Atemayar_Qelisayer', 'Atlantean', 'Aurek-Besh', 'Avesta', 'Ge_ez', 'Glagolitic', 'Gurmukhi', 'Kannada', 'Keble', 'Malayalam', 'Manipuri', 'Mongolian', 'Old_Church_Slavonic_(Cyrillic)', 'Oriya', 'Sylheti', 'Syriac_(Serto)', 'Tengwar', 'Tibetan', 'ULOG']
            self.character_nums = [20, 26, 26, 26, 26, 26, 45, 45, 41, 26, 47, 40, 30, 45, 46, 28, 23, 25, 42, 26]

        else: # all splits
            self.alphabet_names = ['Alphabet_of_the_Magi', 'Anglo-Saxon_Futhorc', 'Arcadian', 'Armenian', 'Asomtavruli_(Georgian)', 'Balinese', 'Bengali', 'Blackfoot_(Canadian_Aboriginal_Syllabics)', 'Braille', 'Burmese_(Myanmar)', 'Cyrillic', 'Early_Aramaic', 'Futurama', 'Grantha', 'Greek', 'Gujarati', 'Hebrew', 'Inuktitut_(Canadian_Aboriginal_Syllabics)', 'Japanese_(hiragana)', 'Japanese_(katakana)', 'Korean', 'Latin', 'Malay_(Jawi_-_Arabic)', 'Mkhedruli_(Georgian)', 'N_Ko', 'Ojibwe_(Canadian_Aboriginal_Syllabics)', 'Sanskrit', 'Syriac_(Estrangelo)', 'Tagalog', 'Tifinagh', 'Angelic', 'Atemayar_Qelisayer', 'Atlantean', 'Aurek-Besh', 'Avesta', 'Ge_ez', 'Glagolitic', 'Gurmukhi', 'Kannada', 'Keble', 'Malayalam', 'Manipuri', 'Mongolian', 'Old_Church_Slavonic_(Cyrillic)', 'Oriya', 'Sylheti', 'Syriac_(Serto)', 'Tengwar', 'Tibetan', 'ULOG']
            self.character_nums = [20, 29, 26, 41, 40, 24, 46, 14, 26, 34, 33, 22, 26, 43, 24, 48, 22, 16, 52, 47, 40, 26, 40, 41, 33, 14, 42, 23, 17, 55, 20, 26, 26, 26, 26, 26, 45, 45, 41, 26, 47, 40, 30, 45, 46, 28, 23, 25, 42, 26]

        self.n_images = sum(self.character_nums) * 20

    def __len__(self):
        return self.n_images

    def get_single_image(self, idx):
        j = idx % 20
        i = idx // 20
        label = i

        for k in range(len(self.alphabet_names)):
            if(i < self.character_nums[k]):
                alphabet = self.alphabet_names[k]+"/"
                if(self.split=="train"):
                    folder = "images_background/"
                elif(self.split=="test"):
                    folder = "images_evaluation/"
                else:
                    folder = "images_background/" if k<30 else "images_evaluation/"
                break
            else:
                i -= self.character_nums[k]

        # char_name = "omniglot/"+folder+alphabet+"character"+(str(i+1) if i>8 else "0"+str(i+1))+"/"
        char_name = "./omniglot/"+folder+alphabet+"character"+(str(i+1) if i>8 else "0"+str(i+1))+"/"
        example_list = sorted(os.listdir(char_name))
        img = Image.open(char_name+example_list[j])

        return self.transform(img.convert('L'))

    def __getitem__(self, idx):
        j = idx % 20
        prange = list(range(idx-j, idx)) + list(range(idx+1, idx-j+20))
        nrange = list(range(idx-j)) + list(range(idx-j+20,self.n_images))
        idx_p = random.choice(prange)
        idx_n = random.choice(nrange)

        return (self.get_single_image(idx),self.get_single_image(idx_p),self.get_single_image(idx_n))
        
