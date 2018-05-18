# Intelligent-Mobile-Projects-with-TensorFlow
This is the code repository for [Intelligent-Mobile-Projects-with-TensorFlow](https://www.packtpub.com/application-development/intelligent-mobile-projects-tensorflow), published by [Packt](https://www.packtpub.com/?utm_source=github). It contains all the supporting project files necessary to work through the book from start to finish.
## About the Book
As a developer, you always need to keep an eye out and be ready for what will be trending soon, while also focusing on what's trending currently. So, what's better than learning about the integration of the best of both worlds, the present and the future? Artificial Intelligence (AI) is widely regarded as the next big thing after mobile, and Google's TensorFlow is the leading open source machine learning framework, the hottest branch of AI. This book covers more than 10 complete iOS, Android, and Raspberry Pi apps powered by TensorFlow and built from scratch, running all kinds of cool TensorFlow models offline on-device: from computer vision, speech and language processing to generative adversarial networks and AlphaZero-like deep reinforcement learning. You’ll learn how to use or retrain existing TensorFlow models, build your own models, and develop intelligent mobile apps running those TensorFlow models. You'll learn how to quickly build such apps with step-by-step tutorials and how to avoid many pitfalls in the process with lots of hard-earned troubleshooting tips.

## Instructions and Navigation
All of the code is organized into folders. Each folder starts with a number followed by the application name. For example, Chapter02.



The code will look like the following:
```
syntax = "proto2";
package object_detection.protos;
message StringIntLabelMapItem {
  optional string name = 1;
  optional int32 id = 2;
  optional string display_name = 3;
};

message StringIntLabelMap {
  repeated StringIntLabelMapItem item = 1;
};
```

We recommend that you start with reading the first four chapters in order, along with running the accompanying iOS and Android apps available from the book's source code repository at http://github.com/jeffxtang/mobiletfbook. That'll help you ensure that you have the development environments all set up for TensorFlow mobile app development and that you know how to integrate TensorFlow into your own iOS and/or Android apps. If you're an iOS developer, you'll also learn how to use Objective-C or Swift with TensorFlow, and when and how to use the TensorFlow pod or the manual TensorFlow iOS library.



Then, if you need to build a custom TensorFlow Android library, go to Chapter 7, Recognizing Drawing with CNN and LSTM, and if you want to learn how to use a Keras model in your mobile app, check out Chapter 8, Predicting Stock Price with RNN, and Chapter 10, Building an AlphaZero-like Mobile Game App.

If you're more interested in TensorFlow Lite or Core ML, read Chapter 11, Using TensorFlow Lite and Core ML on Mobile, and if you're most interested in TensorFlow on Raspberry Pi, or reinforcement learning in TensorFlow, jump to Chapter 12, Developing TensorFlow Apps on Raspberry Pi.

Other than that, you can go through chapters 5 to 10 in order to see how to train different kinds of CNN, RNN, LSTM, GAN, and AlphaZero models and how to use them on mobile, maybe running the iOS and/or Android apps for each chapter before looking into the detailed implementation. Alternatively, you can jump directly to any chapter with the model you're most interested in; just be aware that a later chapter may refer to an earlier chapter for some duplicated details, such as steps of adding a TensorFlow custom iOS library to your iOS app, or fixing some model loading or running errors by building a TensorFlow custom library. However, rest assured that you won't be lost, or at least we've done our best to provide user-friendly and step-by-step tutorials, with occasional references to some steps of previous tutorials, to help you avoid all possible pitfalls you may encounter when building mobile TensorFlow apps, while also avoiding repeating ourselves.

## Related Products
* [Learn QT 5](https://www.packtpub.com/web-development/learn-qt-5?utm_source=github&utm_medium=repository&utm_campaign=9781788478854)

* [Qt 5 Projects](https://www.packtpub.com/application-development/qt-5-projects?utm_source=github&utm_medium=repository&utm_campaign=9781788293884)

* [Game Programming Using Qt: Beginner's Guide](https://www.packtpub.com/game-development/game-programming-using-qt?utm_source=github&utm_medium=repository&utm_campaign=9781782168874)
