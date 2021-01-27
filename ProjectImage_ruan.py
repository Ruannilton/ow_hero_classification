import os
import shutil
import tensorflow
from tensorflow.keras import models
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset = './data/'
pathname = './heroes/'

# Prepara lista de tags assumindo que os arquivos estao em pastas nomeadas


def get_taglist(path="./heroes/"):
    names = [x[0] for x in os.walk(path)]
    names = [n[len(path):] for n in names]
    names = [n for n in names if n]

    return names


tags = get_taglist()

# Prepara pastas de treino, validacao e teste


def prepare_folders(path="./heroes"):
    # Cria pasta 'dataset'
    if not os.path.exists(dataset):
        print(f"step1 creating dataset folder")
        os.mkdir(dataset)

    # Cria pasta 'dataset/train'
    train_dir = os.path.join(dataset, 'train')
    if not os.path.exists(train_dir):
        print(f" step2 creating {train_dir}")
        os.mkdir(train_dir)

    # Cria pasta 'dataset/valid'
    valid_dir = os.path.join(dataset, 'valid')
    if not os.path.exists(valid_dir):
        print(f" step3 creating {valid_dir}")
        os.mkdir(valid_dir)

    # Cria pasta 'dataset/test'
    test_dir = os.path.join(dataset, 'test')
    if not os.path.exists(test_dir):
        print(f" step4 creating {test_dir}")
        os.mkdir(test_dir)

    # Cria subpastas de 'dataset/train'
    named_train_dirs = {}
    for tag in tags:
        name = os.path.join(train_dir, tag)
        print(name)
        named_train_dirs[tag] = name

    for p in named_train_dirs:
        if not os.path.exists(named_train_dirs[p]):
            print(f"creating {named_train_dirs[p]}")
            os.mkdir(named_train_dirs[p])

    # Cria subpastas de 'dataset/valid'
    named_valid_dirs = {}
    for tag in tags:
        name = os.path.join(valid_dir, tag)
        print(name)
        named_valid_dirs[tag] = name

    for p in named_valid_dirs:
        if not os.path.exists(named_valid_dirs[p]):
            print(f"creating {named_valid_dirs[p]}")
            os.mkdir(named_valid_dirs[p])

    # Cria subpastas de 'dataset/test'
    named_test_dirs = {}
    for tag in tags:
        name = os.path.join(test_dir, tag)
        print(name)
        named_test_dirs[tag] = os.path.join(test_dir, tag)

        for p in named_test_dirs:
            if not os.path.exists(named_test_dirs[p]):
                print(f"creating {named_test_dirs[p]}")
                os.mkdir(named_test_dirs[p])

    return (train_dir, valid_dir, test_dir, named_train_dirs, named_valid_dirs, named_test_dirs)


train_dir, valid_dir, test_dir, named_train_dirs, named_valid_dirs, named_test_dirs = prepare_folders()

# copia os dados para pasta de treino
for tag in tags:
    print(f"In {tag}")
    match = [f for f in os.listdir(pathname+f"{tag}") if f[0].isdigit()]
    train_fac = 0.5
    match_length = int(len(match)*train_fac)
    print(match_length)
    for i in range(match_length):
        src = os.path.join(pathname+f"{tag}", match[i])
        dst = os.path.join(named_train_dirs[tag], match[i])
        print(f"copying from {src} to {dst}")
        shutil.copyfile(src, dst)
    print("Done")

 # copia os dados para pasta de validacao
for tag in tags:
    print(f"In {tag}")
    match = [f for f in os.listdir(pathname+f"{tag}") if f[0].isdigit()]
    train_fac = 0.5
    valid_fac = 0.3
    train_length = int(len(match)*train_fac)
    valid_length = train_length + int(len(match)*valid_fac)
    for i in range(train_length, valid_length, 1):
        src = os.path.join(pathname+f"{tag}", match[i])
        dst = os.path.join(named_valid_dirs[tag], match[i])
        print(f"copying from {src} to {dst}")
        shutil.copyfile(src, dst)
    print("Done")

# copia os dados para pasta de testes
for tag in tags:
    if tag != ".ipynb_checkpoints":
        print(f"In {tag}")
        match = [f for f in os.listdir(pathname+f"{tag}") if f[0].isdigit()]
        train_fac = 0.5
        valid_fac = 0.3
        test_fac = 0.2
        train_length = int(len(match)*train_fac)
        valid_length = train_length + int(len(match)*valid_fac)
        test_length = train_length + valid_length + int(len(match)*test_fac)
        for i in range(valid_length, len(match)):
            src = os.path.join(pathname+f"{tag}", match[i])
            dst = os.path.join(named_test_dirs[tag], match[i])
            print(f"copying from {src} to {dst}")
            shutil.copyfile(src, dst)
        print("Done")

# Configuracoes para KerasImageDataGenerator
DataGeneratorParams = {}
DataGeneratorParams["rescale"] = 1./255
DataGeneratorParams["rotation_range"] = 90
DataGeneratorParams["width_shift_range"] = 0.3
DataGeneratorParams["height_shift_range"] = 0.3
DataGeneratorParams["shear_range"] = 0.3
DataGeneratorParams["zoom_range"] = 0.3
DataGeneratorParams["horizontal_flip"] = True

# training
train_datagen = ImageDataGenerator(**DataGeneratorParams)
# validation
valid_datagen = ImageDataGenerator(**DataGeneratorParams)
# test
test_datagen = ImageDataGenerator(**DataGeneratorParams)

# Configuracoes para datagen
datagen = {}
datagen["target_size"] = (512, 512)
datagen["batch_size"] = 10
datagen["classes"] = tags

train_generator = train_datagen.flow_from_directory(train_dir,  **datagen)
valid_generator = valid_datagen.flow_from_directory(valid_dir, **datagen)
test_generator = test_datagen.flow_from_directory(test_dir, **datagen)


def build_network(input_shape, num_classes):
    model = models.Sequential()
    model.add(Conv2D(filters=64,
                     kernel_size=(7, 7),
                     padding="same", activation="relu", input_shape=input_shape))
    model.add(Conv2D(filters=32,
                     kernel_size=(5, 5),
                     padding="same", activation="relu"))
    model.add(Conv2D(filters=16,
                     kernel_size=(5, 5),
                     padding="same", activation="relu"))
    model.add(Flatten())
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')

if physical_devices:
    for dev in physical_devices:
        tf.config.experimental.set_memory_growth(dev, True)

dt = len(tensorflow.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", dt)
print("tf version: ", tensorflow.__version__)
device = ""
if dt < 1:
    print("No GPU Available, reverting to cpu")
    device = "cpu"
else:
    device = "gpu"

print(f"device: {device}, enabled")

INPUT_SHAPE = (512, 512, 3)
NUM_CLASSES = 30
model = build_network(INPUT_SHAPE, NUM_CLASSES)

history = model.fit(train_generator, steps_per_epoch=10, epochs=30,
                    validation_data=valid_generator, validation_steps=10)
