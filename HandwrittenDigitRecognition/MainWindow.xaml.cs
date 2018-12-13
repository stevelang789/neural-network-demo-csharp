using System.Windows;

namespace HandwrittenDigitRecognition
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private async void MainWindow_OnLoaded(object sender, RoutedEventArgs e)
        {
            var processor = new Recognizer();
            await processor.Run(RootGrid.Children.Add);
        }
    }
}
