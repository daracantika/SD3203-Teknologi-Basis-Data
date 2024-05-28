---

Dara Cantika Dewi\
121450127\
RB\
Three Ways of Storing and Accessing Lots of Images in Python\
<https://realpython.com/storing-images-in-python/>

================================================================

Artikel ini menjelaskan pentingnya memahami berbagai teknik penyimpanan dan pengaksesan gambar di Python. Walaupun menyimpan gambar langsung ke disk dalam format .png atau .jpg sudah cukup untuk kebutuhan sederhana, ada beberapa situasi yang memerlukan metode yang lebih efisien.

- Dataset besar: Saat menangani dataset gambar yang sangat besar, seperti jutaan gambar, menyimpan semuanya langsung ke disk menjadi tidak efisien.
- Pelatihan model pembelajaran mesin:Untuk melatih model pembelajaran mesin seperti Convolutional Neural Networks (CNN), diperlukan dataset yang besar. Memuat seluruh gambar ke dalam memori bisa memakan waktu lama, terutama jika proses pelatihan harus diulang berkali-kali.

Artikel ini membahas tiga metode yang lebih efisien untuk menyimpan dan mengakses gambar:

1. Menyimpan gambar di disk sebagai file .png,
2. Lightning Memory-Mapped Databases (LMDB),
3. Hierarchical Data Format (HDF5).

Sebagai contoh, artikel ini menggunakan dataset gambar dari Canadian Institute for Advanced Research, yang dikenal sebagai CIFAR-10. Dataset ini terdiri dari 60.000 gambar berwarna dengan ukuran 32x32 piksel. Data dapat diunduh dari tautan yang disebutkan dalam artikel. Data yang diunduh bukan file gambar yang bisa dibaca langsung oleh manusia, sehingga diperlukan modul pickle untuk membacanya.

```python
import numpy as np
import pickle
from pathlib import Path

# Path to the unzipped CIFAR data
data_dir = Path("C:/Users/ASUS/Downloads/cifar-10-python/cifar-10-batches-py")

# Unpickle function provided by the CIFAR hosts
def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

images, labels = [], []
for batch in data_dir.glob("data_batch_*"):
    batch_data = unpickle(batch)
    for i, flat_im in enumerate(batch_data[b"data"]):
        im_channels = []
        # Each image is flattened, with channels in order of R, G, B
        for j in range(3):
            im_channels.append(
                flat_im[j * 1024 : (j + 1) * 1024].reshape((32, 32))
            )
        # Reconstruct the original image
        images.append(np.dstack((im_channels)))
        # Save the label
        labels.append(batch_data[b"labels"][i])

print("Loaded CIFAR-10 training set:")
print(f" - np.shape(images)     {np.shape(images)}")
print(f" - np.shape(labels)     {np.shape(labels)}")
```

Output :
Loaded CIFAR-10 training set:

- np.shape(images) (50000, 32, 32, 3)
- np.shape(labels) (50000,)

# Setup for Storing Images on Disk

Pada setup akan dilakukan pengaturan image on disk untuk metode defaul ssaat menyimpan dan mengakses gambar-gambar dari disk dengan penginstallan modul Pillow.

```python
pip install Pillow
```

# LMDB

LMDB (Lightning Memory-Mapped Database) adalah penyimpanan key-value yang cepat dan efisien. Dalam implementasi, LMDB adalah pohon B+ sebagai struktur grafik data seperti pohon yang disimpan dalam memori. LMDB dioptimalkan untuk akses memori yang cepat dengan mengatur komponen kunci sesuai dengan ukuran halaman sistem operasi. Keefisienan LMDB dilakukan dengan memetakan memori secara langsung.

```python
pip install lmdb
```

## HDF5

HDF5 (Hierarchical Data Format) adalah format file yang digunakan untuk menyimpan dataset multidimensi dan grup dalam hierarki dengan file HDF terdiri dari jenis objek dataset dan kelompok. HDF5 memungkinkan penyimpanan array multidimensi dari berbagai ukuran dan tipe data dalam dataset, serta dapat menyimpan grup yang berisi dataset atau grup lainnya.

```python
pip install h5py
```

# Storing a single image

Dilakukan perbandingan kinerja kuantitatif dalam membaca dan menulis file serta penggunaan memori disk pada berbagai jumlah file dengan faktor 10 dari satu gambar ke 100.000 gambar antara tiga metode penyimpanan :

- menggunakan file .png
- menggunakan LMDB
- menggunakan HDF5

Buat folder untuk setiap metode penyimpanan yang berisi semua file database atau gambar, dengan menggunakan modul 'path'.

```python
from pathlib import Path

disk_dir = Path("data/disk/")
lmdb_dir = Path("data/lmdb/")
hdf5_dir = Path("data/hdf5/")
```

```python
disk_dir.mkdir(parents=True, exist_ok=True)
lmdb_dir.mkdir(parents=True, exist_ok=True)
hdf5_dir.mkdir(parents=True, exist_ok=True)
```

## Storing to disk

Fungsi store_single_disk untuk menyimpan gambar sebagai file .png ke disk dengan menggunakan paket Pillow. Metadata gambar, seperti label, disimpan dalam file terpisah dalam format .csv. Penyimpanan label dalam file terpisah, memungkinkan manipulasi label secara independen tanpa harus memuat ulang gambar-gambar.

```python
from PIL import Image
import csv

def store_single_disk(image, image_id, label):
    """ Stores a single image as a .png file on disk.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    Image.fromarray(image).save(disk_dir / f"{image_id}.png")

    with open(disk_dir / f"{image_id}.csv", "wt") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow([label])
```

## Storing to LMDB

LMDB (Lightning Memory-Mapped Database) adalah sistem penyimpanan key value, di mana setiap entri disimpan sebagai array byte. Untuk menyimpan gambar ke LMDB, gambar diubah menjadi byte array dan disimpan bersama dengan labelnya. Penggunaan pickle untuk serialisasi penyimpanan objek Python dalam bentuk string.
Penyimpanan label gambar secara terpisah dari data gambar memungkinkan manipulasi label secara independen tanpa harus memuat ulang gambar-gambar. Penggunaan LMDB cocok untuk menyimpan data dalam jumlah besar dengan akses cepat dan efisien.

```python
class CIFAR_Image:
    def __init__(self, image, label):
        # Dimensions of image for reconstruction - not really necessary
        # for this dataset, but some datasets may include images of
        # varying sizes
        self.channels = image.shape[2]
        self.size = image.shape[:2]

        self.image = image.tobytes()
        self.label = label

    def get_image(self):
        """ Returns the image as a numpy array. """
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)
```

```python
import lmdb
import pickle

def store_single_lmdb(image, image_id, label):
    """ Stores a single image to a LMDB.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    map_size = image.nbytes * 10

    # Create a new LMDB environment
    env = lmdb.open(str(lmdb_dir / f"single_lmdb"), map_size=map_size)

    # Start a new write transaction
    with env.begin(write=True) as txn:
        # All key-value pairs need to be strings
        value = CIFAR_Image(image, label)
        key = f"{image_id:08}"
        txn.put(key.encode("ascii"), pickle.dumps(value))
    env.close()
```

## Storing with HDF5

Untuk melakukan penyimapanan gambar ke file HDF5, akan dibuat dalam dua dataset untuk gambar dan metadatanya. Penggunaan modul h5py memungkinkan pembuatan dataset dengan mudah dalam file HDF5.

```python
import h5py

def store_single_hdf5(image, image_id, label):
    """ Stores a single image to an HDF5 file.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "image", np.shape(image), h5py.h5t.STD_U8BE, data=image
    )
    meta_set = file.create_dataset(
        "meta", np.shape(label), h5py.h5t.STD_U8BE, data=label
    )
    file.close()
```

# Experiment for storing a single image

Eksperimen dilakukan untuk menyimpan satu gambar menggunakan tiga metode yang berbeda yaitu disk, LMDB, dan HDF5. Hasil eksperimen menunjukkan waktu yang dibutuhkan untuk menyimpan satu gambar beserta metadatanya, serta penggunaan memori yang terkait dengan setiap metode.

```python
_store_single_funcs = dict(
    disk=store_single_disk, lmdb=store_single_lmdb, hdf5=store_single_hdf5
)
```

```python
from timeit import timeit

store_single_timings = dict()

for method in ("disk", "lmdb", "hdf5"):
    t = timeit(
        "_store_single_funcs[method](image, 0, label)",
        setup="image=images[0]; label=labels[0]",
        number=1,
        globals=globals(),
    )
    store_single_timings[method] = t
    print(f"Method: {method}, Time usage: {t}")
```

output:
Method: disk, Time usage: 0.04620769992470741
Method: lmdb, Time usage: 0.021540199988521636
Method: hdf5, Time usage: 0.012781399884261191

Berdasarkan hasil output dapat diketahui bahwa waktu penyimpanan untuk satu gambar dengan metode disk lebih cepat dibandingkan metode LMDB dan HDF5. Namun ketiga metode tersebut menunjukan waktu penyimpanan yang sangat cepat dalam proses penyimpanan.

# Storing Many Images

Berikut fungsi/code untuk menerima banyak gambar dengan metode disk, LMDB, dan HDF5.

```python
def store_many_disk(images, labels):
    """ Stores an array of images to disk
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    # Save all the images one by one
    for i, image in enumerate(images):
        Image.fromarray(image).save(disk_dir / f"{i}.png")

    # Save all the labels to the csv file
    with open(disk_dir / f"{num_images}.csv", "w") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for label in labels:
            # This typically would be more than just one value per row
            writer.writerow([label])

def store_many_lmdb(images, labels):
    """ Stores an array of images to LMDB.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    map_size = num_images * images[0].nbytes * 10

    # Create a new LMDB DB for all the images
    env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"), map_size=map_size)

    # Same as before — but let's write all the images in a single transaction
    with env.begin(write=True) as txn:
        for i in range(num_images):
            # All key-value pairs need to be Strings
            value = CIFAR_Image(images[i], labels[i])
            key = f"{i:08}"
            txn.put(key.encode("ascii"), pickle.dumps(value))
    env.close()

def store_many_hdf5(images, labels):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    meta_set = file.create_dataset(
        "meta", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
    )
    file.close()
```

# Preparating the datasets

Persiapan dataset untuk melakukan pengujian dengan menggandakan ukuran dataset untuk menguji hingga 100.000 gambar.

```python
cutoffs = [10, 100, 1000, 10000, 100000]

# Let's double our images so that we have 100,000
images = np.concatenate((images, images), axis=0)
labels = np.concatenate((labels, labels), axis=0)

# Make sure you actually have 100,000 images and labels
print(np.shape(images))
print(np.shape(labels))
```

output :
(100000, 32, 32, 3)
(100000,)

Datasets berjumlah 100000 dengan ukuran 32 x 32 dan format warna RGB.

# Experiment for many images

Eksperimen dilakukan dengan menyimpan gambar yang banyak menggunakan metode disk, LMDB, dan HDF5. Hasil eksperimen akan ditunjukan menggunakan visualisasi grafik untuk membandingkan waktu penyimpanan gambar dengan berbagai metode.

```python
_store_many_funcs = dict(
    disk=store_many_disk, lmdb=store_many_lmdb, hdf5=store_many_hdf5
)

from timeit import timeit

store_many_timings = {"disk": [], "lmdb": [], "hdf5": []}

for cutoff in cutoffs:
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_store_many_funcs[method](images_, labels_)",
            setup="images_=images[:cutoff]; labels_=labels[:cutoff]",
            number=1,
            globals=globals(),
        )
        store_many_timings[method].append(t)

        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, Time usage: {t}")
```

output :
Method: disk, Time usage: 0.010568999918177724
Method: lmdb, Time usage: 0.005193699966184795
Method: hdf5, Time usage: 0.004782300093211234
Method: disk, Time usage: 0.09530399995855987
Method: lmdb, Time usage: 0.006910299998708069
Method: hdf5, Time usage: 0.004275999963283539
Method: disk, Time usage: 1.1795103000476956
Method: lmdb, Time usage: 0.08784239995293319
Method: hdf5, Time usage: 0.006575599894858897
Method: disk, Time usage: 12.914711400051601
Method: lmdb, Time usage: 0.5481161000207067
Method: hdf5, Time usage: 0.155798300052993
Method: disk, Time usage: 107.09133470000234
Method: lmdb, Time usage: 3.167688100016676
Method: hdf5, Time usage: 0.4988666999852285

Setiap metode penyimpanan akan menyimpan gambar sebayak 111.110 dalam tiga kali format berbeda. Berdasarkan waktu yang dihasilkan dalam pembacaan banyak gambar, metode HDF5 lebih cepat, sedangkan metode disk memerlukan waktu yang cukup lama. Sehingga, metode HDF5 dan LMDB lebih cocok untuk penyimpanan data dalam jumlah besar.

```python
import matplotlib.pyplot as plt

def plot_with_legend(
    x_range, y_data, legend_labels, x_label, y_label, title, log=False
):
    """ Displays a single plot with multiple datasets and matching legends.
        Parameters:
        --------------
        x_range         list of lists containing x data
        y_data          list of lists containing y values
        legend_labels   list of string legend labels
        x_label         x axis label
        y_label         y axis label
    """
    plt.style.use("seaborn-whitegrid")
    plt.figure(figsize=(10, 7))

    if len(y_data) != len(legend_labels):
        raise TypeError(
            "Error: number of data sets does not match number of labels."
        )

    all_plots = []
    for data, label in zip(y_data, legend_labels):
        if log:
            temp, = plt.loglog(x_range, data, label=label)
        else:
            temp, = plt.plot(x_range, data, label=label)
        all_plots.append(temp)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(handles=all_plots)
    plt.show()

# Getting the store timings data to display
disk_x = store_many_timings["disk"]
lmdb_x = store_many_timings["lmdb"]
hdf5_x = store_many_timings["hdf5"]

plot_with_legend(
    cutoffs,
    [disk_x, lmdb_x, hdf5_x],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to store",
    "Storage time",
    log=False,
)

plot_with_legend(
    cutoffs,
    [disk_x, lmdb_x, hdf5_x],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to store",
    "Log storage time",
    log=True,
)
```

![alt text](gambar1.png)
![alt text](gambar2.png)

Pada grafik pertama menunjukkan waktu penyimpanan normal yang tidak disesuaikan, menyoroti perbedaan drastis antara penyimpanan ke file dan LMDB atau HDF5 .png. Sedangkan grafik kedua menunjukkan pengaturan waktu, menyoroti bahwa HDF5 dimulai lebih lambat dari LMDB tetapi, dengan jumlah gambar yang lebih besar, namun metode HDF5 tetap lebih unggul dibandingkan LMDB

# Reading from Disk

Selanjutnya, dilakukan membaca kembali gambar tunggal beserta meta data yang sebelumnya disimpan dengan tiga metode yang berbeda : disk, LMDB, dan HDF5.

## Experiment for reading as single image

### Membaca data dari disk

```python
def read_single_disk(image_id):
    """ Stores a single image to disk.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    image = np.array(Image.open(disk_dir / f"{image_id}.png"))

    with open(disk_dir / f"{image_id}.csv", "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        label = int(next(reader)[0])

    return image, label
```

## Membaca data dari lmdb

```python
def read_single_lmdb(image_id):
    """ Stores a single image to LMDB.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    # Open the LMDB environment
    env = lmdb.open(str(lmdb_dir / f"single_lmdb"), readonly=True)

    # Start a new read transaction
    with env.begin() as txn:
        # Encode the key the same way as we stored it
        data = txn.get(f"{image_id:08}".encode("ascii"))
        # Remember it's a CIFAR_Image object that is loaded
        cifar_image = pickle.loads(data)
        # Retrieve the relevant bits
        image = cifar_image.get_image()
        label = cifar_image.label
    env.close()

    return image, label
```

### Membaca data dari hdf5

```python
def read_single_hdf5(image_id):
    """ Stores a single image to HDF5.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "r+")

    image = np.array(file["/image"]).astype("uint8")
    label = int(np.array(file["/meta"]).astype("uint8"))

    return image, label
```

```python
_read_single_funcs = dict(
    disk=read_single_disk, lmdb=read_single_lmdb, hdf5=read_single_hdf5
)
```

Ketiga metode memungkinkan dalam pembacaan kembali gambar tunggal beserta metadata. Membaca disk merupakan yang paling sederhana, sedangkan membaca dari LMDB membutuhkan proses deserialisasi tambahan, sedangkan membaca HDF5 mirip dengan proses mennulis, dimana dataset diakses menggunakan pengindeksan pada objek file.

## Experiment for Reading a Single Image

```python
from timeit import timeit

read_single_timings = dict()

for method in ("disk", "lmdb", "hdf5"):
    t = timeit(
        "_read_single_funcs[method](0)",
        setup="image=images[0]; label=labels[0]",
        number=1,
        globals=globals(),
    )
    read_single_timings[method] = t
    print(f"Method: {method}, Time usage: {t}")
```

output:
Method: disk, Time usage: 0.014625300071202219
Method: lmdb, Time usage: 0.022377900080755353
Method: hdf5, Time usage: 0.015335400006733835

Berdasarkan hasil pembacaan data, diketahui bahwa metode LMDB memiliki waktu tercepat diantara ketiga metode.

# Reading many images

## Adjusting the Code for Many Images

Dengan fungsi diatas dilakukan penerapan dalam ketiga metode dengan gambar yang lebih banyak.

```python
def read_many_disk(num_images):
    """ Reads image from disk.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []

    # Loop over all IDs and read each image in one by one
    for image_id in range(num_images):
        images.append(np.array(Image.open(disk_dir / f"{image_id}.png")))

    with open(disk_dir / f"{num_images}.csv", "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for row in reader:
            labels.append(int(row[0]))
    return images, labels

def read_many_lmdb(num_images):
    """ Reads image from LMDB.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []
    env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"), readonly=True)

    # Start a new read transaction
    with env.begin() as txn:
        # Read all images in one single transaction, with one lock
        # We could split this up into multiple transactions if needed
        for image_id in range(num_images):
            data = txn.get(f"{image_id:08}".encode("ascii"))
            # Remember that it's a CIFAR_Image object
            # that is stored as the value
            cifar_image = pickle.loads(data)
            # Retrieve the relevant bits
            images.append(cifar_image.get_image())
            labels.append(cifar_image.label)
    env.close()
    return images, labels

def read_many_hdf5(num_images):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []

    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "r+")

    images = np.array(file["/images"]).astype("uint8")
    labels = np.array(file["/meta"]).astype("uint8")

    return images, labels

_read_many_funcs = dict(
    disk=read_many_disk, lmdb=read_many_lmdb, hdf5=read_many_hdf5
)
```

## experiment for reading many images

```python
from timeit import timeit

read_many_timings = {"disk": [], "lmdb": [], "hdf5": []}

for cutoff in cutoffs:
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_read_many_funcs[method](num_images)",
            setup="num_images=cutoff",
            number=0,
            globals=globals(),
        )
        read_many_timings[method].append(t)

        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, No. images: {cutoff}, Time usage: {t}")
```

output:
Method: disk, No. images: 10, Time usage: 2.00001522898674e-07
Method: lmdb, No. images: 10, Time usage: 4.00003045797348e-07
Method: hdf5, No. images: 10, Time usage: 3.00002284348011e-07
Method: disk, No. images: 100, Time usage: 2.00001522898674e-07
Method: lmdb, No. images: 100, Time usage: 3.00002284348011e-07
Method: hdf5, No. images: 100, Time usage: 4.00003045797348e-07
Method: disk, No. images: 1000, Time usage: 3.00002284348011e-07
Method: lmdb, No. images: 1000, Time usage: 3.00002284348011e-07
Method: hdf5, No. images: 1000, Time usage: 3.00002284348011e-07
Method: disk, No. images: 10000, Time usage: 2.00001522898674e-07
Method: lmdb, No. images: 10000, Time usage: 2.00001522898674e-07
Method: hdf5, No. images: 10000, Time usage: 2.00001522898674e-07
Method: disk, No. images: 100000, Time usage: 6.00004568696022e-07
Method: lmdb, No. images: 100000, Time usage: 3.00002284348011e-07
Method: hdf5, No. images: 100000, Time usage: 2.00001522898674e-07

Berdasarkan output yang didapatkan, diketahui bahwa waktu penyimpanan gambar relatif konstan untuk setiap metode dengan jumlah gambar yang disimpan bertambah. Hal ini menunjukkan bahwa ketiga metode (disk, LMDB, dan HDF5) mampu menangani penambahan jumlah gambar dengan efisien, tanpa mengalami peningkatan waktu yang signifikan.

```python
disk_x_r = read_many_timings["disk"]
lmdb_x_r = read_many_timings["lmdb"]
hdf5_x_r = read_many_timings["hdf5"]

plot_with_legend(
    cutoffs,
    [disk_x_r, lmdb_x_r, hdf5_x_r],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to read",
    "Read time",
    log=False,
)

plot_with_legend(
    cutoffs,
    [disk_x_r, lmdb_x_r, hdf5_x_r],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to read",
    "Log read time",
    log=True,
)
```

![alt text](gambar3.png)
![alt text](gambar4.png)

Grafik tersebut menunjukkan waktu yang dibutuhkan untuk membaca gambar dengan metode disk, LMDB, dan HDF5. Hasil dari grafik tersebut menunjukkan waktu membaca gambar dengan metode disk meningkat seiring dengan jumlah gambar. Hal ini menunjukkan bahwa membaca gambar PNG menggunakan metode disk menjadi tidak efisien untuk dataset besar. Sedangkan, waktu baca untuk LMDB dan HDF5 relatif stabil, bahkan untuk dataset besar. Hal ini menunjukkan bahwa kedua metode ini lebih efisien untuk penyimpanan dan pengaksesan gambar.

```python
plot_with_legend(
    cutoffs,
    [disk_x_r, lmdb_x_r, hdf5_x_r, disk_x, lmdb_x, hdf5_x],
    [
        "Read PNG",
        "Read LMDB",
        "Read HDF5",
        "Write PNG",
        "Write LMDB",
        "Write HDF5",
    ],
    "Number of images",
    "Seconds",
    "Log Store and Read Times",
    log=False,
)
```

![alt text](gambar5.png)

Grafik tersebut menunjukkan waktu yang dibutuhkan untuk membaca dan menulis gambar dengan metode disk, LMDB, dan HDF5. Hasil dari grafik tersebut menunjukkan waktu menulis gambar dengan metode disk meningkat seiring dengan jumlah gambar. Hal ini menunjukkan bahwa menyimpan gambar PNG di disk menjadi tidak efisien untuk dataset besar. Sedangkan, waktu baca untuk LMDB dan HDF5 relatif stabil, bahkan untuk dataset besar. Hal ini menunjukkan bahwa kedua metode ini lebih efisien untuk penyimpanan dan pengaksesan gambar.

```python
# Memory used in KB
disk_mem = [24, 204, 2004, 20032, 200296]
lmdb_mem = [60, 420, 4000, 39000, 393000]
hdf5_mem = [36, 304, 2900, 29000, 293000]

X = [disk_mem, lmdb_mem, hdf5_mem]

ind = np.arange(3)
width = 0.35

plt.subplots(figsize=(8, 10))
plots = [plt.bar(ind, [row[0] for row in X], width)]
for i in range(1, len(cutoffs)):
    plots.append(
        plt.bar(
            ind, [row[i] for row in X], width, bottom=[row[i - 1] for row in X]
        )
    )

plt.ylabel("Memory in KB")
plt.title("Disk memory used by method")
plt.xticks(ind, ("PNG", "LMDB", "HDF5"))
plt.yticks(np.arange(0, 400000, 100000))

plt.legend(
    [plot[0] for plot in plots], ("10", "100", "1,000", "10,000", "100,000")
)
plt.show()
```

![alt text](gambar6.png)

Diagram batang ini membandingkan penggunaan memori disk oleh berbagai metode dalam menyimpan dan mengakses gambar, yaitu:

- PNG:Menyimpan gambar sebagai file PNG di disk.
- LMDB: Menyimpan gambar dalam Lightning Memory-Mapped Database.
- HDF5:Menyimpan gambar dalam format Hierarchical Data Format 5.

Berdasarkan hasil yang ditunjukkan oleh diagram batang, penggunaan memori disk oleh file PNG meningkat secara linear seiring dengan bertambahnya jumlah gambar. Sebaliknya, penggunaan memori disk oleh file LMDB dan HDF5 tetap relatif stabil, bahkan untuk dataset besar. Ini menunjukkan bahwa metode LMDB dan HDF5 lebih efisien untuk menyimpan dataset gambar yang besar.
