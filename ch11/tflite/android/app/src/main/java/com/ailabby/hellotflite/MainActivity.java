package com.ailabby.hellotflite;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity implements Runnable {
    private ImageView mImageView;
    private Button mButton;

    private static final String IMG_FILE = "www.jpg";
    private static final int INPUT_SIZE = 224;
    private ImageClassifier classifier;

    private static final String TAG = "HelloTFLite";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mImageView = findViewById(R.id.imageview);
        try {
            AssetManager am = getAssets();
            InputStream is = am.open(IMG_FILE);
            Bitmap bitmap = BitmapFactory.decodeStream(is);
            mImageView.setImageBitmap(bitmap);
        } catch (IOException e) {
            e.printStackTrace();
        }

        mButton = findViewById(R.id.button);
        mButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mButton.setText("Classifying...");
                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });
    }


    @Override
    public void onStart() {
        super.onStart();
        try {
            classifier = new ImageClassifier(this);
        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to initialize an image classifier.");
        }

        if (classifier == null ) {
            Log.e(TAG, "Uninitialized Classifier or invalid context.");
            return;
        }
    }



    @Override
    public void run() {

        try {
            Bitmap bitmap = BitmapFactory.decodeStream(getAssets().open(IMG_FILE));
            Bitmap croppedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
            String result = classifier.classifyFrame(croppedBitmap);
            Log.d(TAG,  result);
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }
}
