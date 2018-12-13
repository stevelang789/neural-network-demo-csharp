using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Forms.Integration;
using System.Windows.Media;
using System.Windows.Shapes;
using System.Windows.Threading;
using NeuralNetwork;

namespace HandwrittenDigitRecognition
{
    internal class Visualizer
    {
        public const int ImageWidthHeight = 28;

        public static Panel NeuralNetRenderer(Trainer trainer, bool centreImage)
        {
            const int maxItemsToDisplay = 100;

            var net = trainer.Network;
            var firingNetwork = new FiringNetwork(net);
            var canvas = new Canvas { Margin = new Thickness(10) };

            var children = (
                from layer in net.Neurons
                from neuron in layer.Take(maxItemsToDisplay)
                select new
                {
                    neuron,
                    layer,
                    circle = new Border
                    {
                        Background = new SolidColorBrush(Color.FromRgb(200, 200, 200)),
                        CornerRadius = new CornerRadius(50),
                        Child = neuron.IsOutputNeuron ? new TextBlock
                        {
                            Text = neuron.Index.ToString(),
                            HorizontalAlignment = HorizontalAlignment.Center,
                            VerticalAlignment = VerticalAlignment.Center,
                            FontSize = 15,
                            FontWeight = FontWeights.Bold,
                            Foreground = Brushes.White
                        } : null
                    },
                    lines = neuron.InputWeights.Take(neuron.Layer == 0 ? 0 : maxItemsToDisplay).Select((weight, i) => new
                    {
                        Index = i,
                        Line = new Line
                        {
                            Stroke = new SolidColorBrush(Color.FromArgb(64, 80, 80, 80)),
                            StrokeThickness = 3,
                            StrokeStartLineCap = PenLineCap.Round,
                            StrokeEndLineCap = PenLineCap.Round
                        }
                    }).ToArray()
                }).ToArray();

            double GetDiameter(Array layer) => canvas.ActualHeight / (Math.Min(maxItemsToDisplay, layer.Length) + 3);

            canvas.SizeChanged += delegate
            {
                var layer = children.First().layer;
                var layerIndex = 0;
                foreach (var n in children)
                {
                    if (layer != n.layer) layerIndex++;
                    layer = n.layer;
                    n.circle.Width = n.circle.Height = GetDiameter(layer);
                    var xPerItem = (canvas.ActualWidth - GetDiameter(layer)) / (net.Neurons.Length - 1);
                    var left = xPerItem * layerIndex;
                    var top = canvas.ActualHeight / Math.Min(maxItemsToDisplay, layer.Length) * n.neuron.Index + n.circle.Height / 2;
                    n.circle.SetValue(Canvas.LeftProperty, left);
                    n.circle.SetValue(Canvas.TopProperty, top);
                    var i = 0;
                    foreach (var l in n.lines)
                    {
                        var prevLayer = net.Neurons[n.neuron.Layer - 1];
                        l.Line.X1 = left - xPerItem + GetDiameter(prevLayer) / 2;
                        l.Line.X2 = left + GetDiameter(layer) / 2;
                        l.Line.Y1 = canvas.ActualHeight / Math.Min(maxItemsToDisplay, prevLayer.Length) * i + GetDiameter(prevLayer);
                        l.Line.Y2 = top + GetDiameter(layer) / 2;
                        i++;
                    }
                }
            };

            var infoPanel = new DockPanel { Margin = new Thickness(10) };
            infoPanel.SetValue(Grid.ColumnProperty, 1);

            var lblTrainingInfo = new Label { FontSize = 18, Margin = new Thickness(0, 0, 0, 10) };
            lblTrainingInfo.SetValue(DockPanel.DockProperty, Dock.Top);
            infoPanel.Children.Add(lblTrainingInfo);

            var lblLiveTraining = new Label { FontSize = 18, Foreground = Brushes.Green };
            lblLiveTraining.SetValue(DockPanel.DockProperty, Dock.Top);
            infoPanel.Children.Add(lblLiveTraining);

            var lblMessage = new Label { FontSize = 18, Foreground = Brushes.Blue };
            lblMessage.SetValue(DockPanel.DockProperty, Dock.Top);
            infoPanel.Children.Add(lblMessage);

            var btnClear = new Button { Content = "Clear >>", Padding = new Thickness(10) };
            btnClear.SetValue(DockPanel.DockProperty, Dock.Bottom);
            infoPanel.Children.Add(btnClear);

            var lblPrediction = new Label { FontSize = 100, Margin = new Thickness(0), FontWeight = FontWeights.Bold, Foreground = Brushes.Blue, HorizontalContentAlignment = HorizontalAlignment.Center };
            infoPanel.Children.Add(lblPrediction);

            var lastIterations = 0;
            var timer = new DispatcherTimer { IsEnabled = true, Interval = TimeSpan.FromMilliseconds(150) };
            timer.Tick += delegate
            {
                lblTrainingInfo.Content = trainer.TrainingInfo;
                lblMessage.Content = trainer.Message;

                if (trainer.CurrentEpoch > 0)
                    lblLiveTraining.Content = "Current epoch = " + trainer.CurrentEpoch +
                    (trainer.CurrentAccuracy == 0 ? "" : "\r\nLast training score = " + (trainer.CurrentAccuracy).ToString("N1") + "%");

                if (trainer.Iterations == lastIterations) return;
                lastIterations = trainer.Iterations;

                var minMax = children.GroupBy(c => c.neuron.Layer).ToDictionary(
                    g => g.Key,
                    g => new
                    {
                        MinBias = Math.Min(-1, g.Min(x => x.neuron.Bias)),
                        MaxBias = Math.Max(1, g.Max(x => x.neuron.Bias)),
                        MinWeight = Math.Min(-1, g.SelectMany(x => x.neuron.InputWeights).Min(l => l)),
                        MaxWeight = Math.Max(1, g.SelectMany(x => x.neuron.InputWeights).Max(l => l)),
                    });

                foreach (var n in children)
                {
                    var minMaxEntry = minMax[n.neuron.Layer];
                    n.circle.Background = new SolidColorBrush(GetColor(
                        n.neuron.Bias,
                        minMaxEntry.MinBias,
                        minMaxEntry.MaxBias,
                        255));
                    n.circle.ToolTip = "Bias=" + n.neuron.Bias;

                    foreach (var l in n.lines)
                    {
                        l.Line.Stroke = new SolidColorBrush(GetColor(
                        n.neuron.InputWeights[l.Index],
                        minMaxEntry.MinWeight,
                        minMaxEntry.MaxWeight,
                        100));
                        l.Line.ToolTip = "Weight=" + n.neuron.InputWeights[l.Index];
                    }
                }
            };

            foreach (var n in children)
                foreach (var l in n.lines)
                    canvas.Children.Add(l.Line);

            foreach (var n in children)
                canvas.Children.Add(n.circle);

            var drawingBox = GetDrawingBox();
            var drawingBoxHost = new WindowsFormsHost
            {
                Child = drawingBox,
                Margin = new Thickness(10),
                HorizontalAlignment = HorizontalAlignment.Right
            };

            drawingBoxHost.SetValue(Grid.ColumnProperty, 2);

            drawingBox.MouseUp += (sender, args) =>
            {
                using (var scaledImage = new System.Drawing.Bitmap(ImageWidthHeight, ImageWidthHeight))
                using (var g = System.Drawing.Graphics.FromImage(scaledImage))
                {
                    g.DrawImage(drawingBox.Image, 0, 0, ImageWidthHeight * drawingBox.Width / drawingBox.Height, ImageWidthHeight);
                    var data = BitmapToByteArray(scaledImage);
                    var greyData = data.SelectMany((b, n) => n % 4 == 1 ? new[] { (byte)(Math.Min(255, b * 3 / 2)) } : new byte[0]).ToArray();
                    if (centreImage) greyData = CentreImage(greyData, ImageWidthHeight);
                    var input = greyData.Select(d => (double)d / 255).ToArray();
                    firingNetwork.FeedForward(input);
                    lblPrediction.Content = Helper.IndexOfMax(firingNetwork.OutputValues.ToArray());
                }
            };

            btnClear.Click += (sender, args) =>
            {
                using (var g = System.Drawing.Graphics.FromImage(drawingBox.Image))
                    g.Clear(System.Drawing.Color.Black);

                drawingBox.Invalidate();

                lblPrediction.Content = "";
            };

            var grid = new Grid();

            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(5, GridUnitType.Star) });
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto, MinWidth = 250 });
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });

            grid.LayoutUpdated += (sender, args) =>
                drawingBoxHost.Width = drawingBoxHost.Height = Math.Max(50, Math.Min(grid.ActualWidth / 3, grid.ActualHeight - 20));

            grid.Children.Add(canvas);
            grid.Children.Add(infoPanel);
            grid.Children.Add(drawingBoxHost);

            var panel = new DockPanel { Background = Brushes.White };
            panel.Children.Add(grid);
            return panel;
        }

        private static Color GetColor(double value, double min, double max, byte alpha)
        {
            if (value < min) value = min;
            if (value > max) value = max;
            var scaledValue = value < 0 ? value / min : value / max;
            byte greyPoint = 200;

            var colorChannel = Convert.ToByte(greyPoint + scaledValue * (255 - greyPoint));
            var secondChannel = Convert.ToByte(greyPoint - scaledValue * greyPoint * 8 / 10);
            var thirdChannel = Convert.ToByte(greyPoint - scaledValue * greyPoint * 9 / 10);

            if (value < 0)
                return Color.FromArgb(alpha, thirdChannel, secondChannel, colorChannel);
            else
                return Color.FromArgb(alpha, colorChannel, secondChannel, thirdChannel);
        }

        private static System.Windows.Forms.PictureBox GetDrawingBox()
        {
            var box = new System.Windows.Forms.PictureBox();
            var pen = new System.Drawing.Pen(System.Drawing.Color.White, System.Windows.Forms.Control.DefaultFont.Height * 2.2f);
            pen.StartCap = pen.EndCap = System.Drawing.Drawing2D.LineCap.Round;
            System.Drawing.Graphics graphics = null;
            box.SizeChanged += delegate
            {
                if (box.Width == 0 || box.Height == 0) return;
                var oldImage = box.Image;
                box.Image = new System.Drawing.Bitmap(box.Width, box.Height);
                oldImage?.Dispose();
                graphics = System.Drawing.Graphics.FromImage(box.Image);
                graphics.FillRectangle(System.Drawing.Brushes.Black, 0, 0, box.Height, box.Height);
            };

            var lastPos = System.Drawing.Point.Empty;
            box.MouseMove += (sender, args) =>
            {
                if (args.Button == System.Windows.Forms.MouseButtons.Left)
                {
                    graphics.DrawLine(pen, lastPos, args.Location);
                    box.Invalidate();
                }
                lastPos = args.Location;
            };

            return box;
        }

        private static byte[] BitmapToByteArray(System.Drawing.Bitmap bitmap)
        {
            System.Drawing.Imaging.BitmapData bmpdata = null;
            try
            {
                bmpdata = bitmap.LockBits(new System.Drawing.Rectangle(0, 0, bitmap.Width, bitmap.Height), System.Drawing.Imaging.ImageLockMode.ReadOnly, bitmap.PixelFormat);
                var numbytes = bmpdata.Stride * bitmap.Height;
                var bytedata = new byte[numbytes];
                var ptr = bmpdata.Scan0;
                Marshal.Copy(ptr, bytedata, 0, numbytes);
                return bytedata;
            }
            finally
            {
                if (bmpdata != null)
                    bitmap.UnlockBits(bmpdata);
            }
        }

        private static byte[] CentreImage(byte[] image, int stride)
        {
            var indexed = image.Select((value, i) => new { Column = i % stride, Row = i / stride, Value = value }).ToArray();
            var orderedX = indexed.Where(x => x.Value > 10).OrderBy(x => x.Column).ToArray();
            if (!orderedX.Any()) return image;
            var leftMargin = orderedX.First().Column;
            var rightMargin = stride - orderedX.Last().Column;
            var orderedY = indexed.Where(x => x.Value > 10).OrderBy(x => x.Row).ToArray();
            var topMargin = orderedY.First().Row;
            var bottomMargin = stride - orderedY.Last().Row;
            var adjustmentRight = (rightMargin - leftMargin) / 2;
            var adjustmentDown = (bottomMargin - topMargin) / 2;
            var newImage = new byte[image.Length];

            for (var i = 0; i < stride; i++)
                for (var j = 0; j < stride; j++)
                {
                    if (i < adjustmentDown || i >= stride + adjustmentDown || j < adjustmentRight || j >= stride + adjustmentRight)
                        newImage[i * stride + j] = 0;
                    else
                        newImage[i * stride + j] = image[(i - adjustmentDown) * stride + j - adjustmentRight];
                }

            return newImage;
        }
    }
}
