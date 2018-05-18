package com.ailabby.mobileai.hellotensorflow;

import android.graphics.Bitmap;
import android.os.SystemClock;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;

import java.io.IOException;
import java.util.List;
import java.util.logging.Logger;

import android.graphics.BitmapFactory;
import android.util.Log;

public class MainActivity extends AppCompatActivity {
    private static final int INPUT_SIZE = 224;
    //private static final int INPUT_SIZE = 299;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128;
    //private static final String INPUT_NAME = "Mul";
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "final_result";

    private static final String MODEL_FILE = "file:///android_asset/dog_retrained_mobilenet10_224_not_quantized.pb";
    //private static final String MODEL_FILE = "file:///android_asset/dog_retrained_mobilenet10_224.pb";
    //private static final String MODEL_FILE = "file:///android_asset/quantized_stripped_dogs_retrained.pb";
    private static final String LABEL_FILE = "file:///android_asset/dog_retrained_labels.txt";
    private static final String IMG_FILE = "lab1.jpg";


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Classifier classifier = TensorFlowImageClassifier.create(
                        getAssets(),
                        MODEL_FILE,
                        LABEL_FILE,
                        INPUT_SIZE,
                        IMAGE_MEAN,
                        IMAGE_STD,
                        INPUT_NAME,
                        OUTPUT_NAME);

        try {
            Bitmap bitmap = BitmapFactory.decodeStream(getAssets().open(IMG_FILE));
            Bitmap croppedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
            final List<Classifier.Recognition> results = classifier.recognizeImage(croppedBitmap);
            for (Classifier.Recognition result : results) {
                Log.d("", result.getTitle() +":" +result.getConfidence());
            }
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }
}
