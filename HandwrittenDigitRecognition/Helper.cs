namespace HandwrittenDigitRecognition
{
    internal class Helper
    {
        public static int IndexOfMax(double[] values)
        {
            double max = 0;
            var indexOfMax = 0;
            for (var i = 0; i < values.Length; i++)
            if (values[i] > max)
            {
                max = values[i];
                indexOfMax = i;
            }

            return indexOfMax;
        }
    }
}
