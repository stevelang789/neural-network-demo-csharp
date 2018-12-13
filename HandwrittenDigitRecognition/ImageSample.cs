using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using NeuralNetwork;

namespace HandwrittenDigitRecognition
{
    internal class ImageSample : Sample
    {
        private const int CategoryCount = 10;

        public byte Label;
        public byte[] Pixels;

        public ImageSample(byte label, byte[] pixels, int categoryCount)
        {
            Label = label;
            Pixels = pixels;
            Data = ToDouble(pixels);
            ExpectedOutput = LabelToDoubleArray(label, categoryCount);
            IsOutputCorrect = input => Helper.IndexOfMax(input) == Label;
        }

        private static double[] ToDouble(byte[] data) => data.Select(p => (double)p / 255).ToArray();

        private static double[] LabelToDoubleArray(byte label, int categoryCount) =>
            Enumerable.Range(0, categoryCount).Select(i => i == label ? 1d : 0).ToArray();

        public static ImageSample[] LoadTrainingImages() =>
            Load(GetDataFilePath("Training Images", TrainingImagesUri), GetDataFilePath("Training Labels", TrainingLabelsUri), CategoryCount);

        public static ImageSample[] LoadTestingImages() =>
            Load(GetDataFilePath("Testing Images", TestingImagesUri), GetDataFilePath("Testing Labels", TestingLabelsUri), CategoryCount);

        private static ImageSample[] Load(string imgPath, string labelPath, int categoryCount)
        {
            var imgData = File.ReadAllBytes(imgPath);
            var header = imgData.Take(16).Reverse().ToArray();
            var rows = BitConverter.ToInt32(header, 4);
            var cols = BitConverter.ToInt32(header, 0);

            return File.ReadAllBytes(labelPath)
                .Skip(8)  // skip header
                .Select((label, i) => new ImageSample(label, SliceArray(imgData, rows * cols * i + header.Length, rows * cols), categoryCount))
                .ToArray();
        }

        private static byte[] SliceArray(byte[] source, int offset, int length)
        {
            var target = new byte[length];
            Array.Copy(source, offset, target, 0, length);
            return target;
        }

        private static readonly string BasePath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "LINQPad Machine Learning",
            "MNIST digits");

        private const string
            TrainingImagesUri = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            TrainingLabelsUri = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            TestingImagesUri = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            TestingLabelsUri = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz";

        private static string GetDataFilePath(string filename, string uri)
        {
            if (!Directory.Exists(BasePath)) Directory.CreateDirectory(BasePath);
            var fullPath = Path.Combine(BasePath, filename);

            if (!File.Exists(fullPath))
            {
                var buffer = new byte[0x10000];
                using (var ms = new MemoryStream(new WebClient().DownloadData(uri)))
                using (var inStream = new GZipStream(ms, CompressionMode.Decompress))
                using (var outStream = File.Create(fullPath))
                    while (true)
                    {
                        var len = inStream.Read(buffer, 0, buffer.Length);
                        if (len == 0) break;
                        outStream.Write(buffer, 0, len);
                    }
            }

            return fullPath;
        }
    }
}
