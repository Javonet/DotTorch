namespace DotnetTorch
{
    using Javonet.Netcore.Sdk;
    using System;

    public class Program
    {
        public static void Main(string[] args)
        {
            Javonet.Activate("paste_your_license_key_here");

            var calledRuntime = Javonet.InMemory().Python();

            // path to custom python code
            var pythonResourcesPath = "absolute_path_to_your_python_code_directory";

            // load custom code
            calledRuntime.LoadLibrary($"{pythonResourcesPath}/scripts");

            // train py model
            var trainClassName = "train.Train";
            var calledTrainRuntime = calledRuntime.GetType(trainClassName).Execute();
            var modelsPath = $"{pythonResourcesPath}/models";
            //calledTrainRuntime.InvokeInstanceMethod("train", calledTrainRuntime, modelsPath).Execute();

            // test py model
            var testClassName = "test_model.TestModel";
            var calledTestRuntime = calledRuntime.GetType(testClassName).Execute();
            calledTestRuntime.InvokeInstanceMethod("test", calledTestRuntime, modelsPath).Execute();

            // test by custom image
            var imagePath = $"{pythonResourcesPath}/images/test_image.jpg";
            var testImageClassName = "test_custom_image.TestCustomImage";
            var calledImageTestRuntime = calledRuntime.GetType(testImageClassName).Execute();
            var predictedClass = (string)calledImageTestRuntime
                .InvokeInstanceMethod("test", calledImageTestRuntime, modelsPath, imagePath)
                .Execute()
                .GetValue();

            Console.WriteLine($"Predicted class by .NET: {predictedClass}");
        }
    }
}
