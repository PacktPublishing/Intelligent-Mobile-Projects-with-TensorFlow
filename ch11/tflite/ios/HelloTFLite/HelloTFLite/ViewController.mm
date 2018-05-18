//
//  ViewController.m
//  HelloTFLite
//
//  Created by Jeff Tang on 3/18/18.
//  Copyright © 2018 Jeff Tang. All rights reserved.
//

#import "ViewController.h"

#include <pthread.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/tools/mutable_op_resolver.h"

#include "ios_image_load.h"

NSString* RunInferenceOnImage();

@interface ViewController ()

@end

@implementation ViewController
-(void) showResult:(NSString *)result {
    UIAlertController* alert = [UIAlertController alertControllerWithTitle:@"TFLite Model Result" message:result preferredStyle:UIAlertControllerStyleAlert];
    
    UIAlertAction* action = [UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:nil];
    [alert addAction:action];
    [self presentViewController:alert animated:YES completion:nil];
}
-(void)tapped:(UITapGestureRecognizer *)tapGestureRecognizer {
        NSString *result = RunInferenceOnImage();
        [self showResult:result];
}

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    UILabel *lbl = [[UILabel alloc] init];
    [lbl setTranslatesAutoresizingMaskIntoConstraints:NO];
    lbl.text = @"Tap Anywhere";
    [self.view addSubview:lbl];
    
    NSLayoutConstraint *horizontal = [NSLayoutConstraint constraintWithItem:lbl
                                                                  attribute:NSLayoutAttributeCenterX
                                                                  relatedBy:NSLayoutRelationEqual
                                                                     toItem:self.view
                                                                  attribute:NSLayoutAttributeCenterX
                                                                 multiplier:1
                                                                   constant:0];
    NSLayoutConstraint *vertical = [NSLayoutConstraint constraintWithItem:lbl
                                                                attribute:NSLayoutAttributeCenterY
                                                                relatedBy:NSLayoutRelationEqual
                                                                   toItem:self.view
                                                                attribute:NSLayoutAttributeCenterY
                                                               multiplier:1
                                                                 constant:0];
    [self.view addConstraint:horizontal];
    [self.view addConstraint:vertical];
    
    UITapGestureRecognizer *recognizer = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(tapped:)];
    [self.view addGestureRecognizer:recognizer];
    
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end


// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
static void GetTopN(const float* prediction, const int prediction_size, const int num_results,
                    const float threshold, std::vector<std::pair<float, int> >* top_results) {
    // Will contain top N results in ascending order.
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int> >,
    std::greater<std::pair<float, int> > >
    top_result_pq;
    
    const long count = prediction_size;
    for (int i = 0; i < count; ++i) {
        const float value = prediction[i];
        
        // Only add it if it beats the threshold and has a chance at being in
        // the top N.
        if (value < threshold) {
            continue;
        }
        
        top_result_pq.push(std::pair<float, int>(value, i));
        
        // If at capacity, kick the smallest value out.
        if (top_result_pq.size() > num_results) {
            top_result_pq.pop();
        }
    }
    
    // Copy to output vector and reverse into descending order.
    while (!top_result_pq.empty()) {
        top_results->push_back(top_result_pq.top());
        top_result_pq.pop();
    }
    std::reverse(top_results->begin(), top_results->end());
}

NSString* FilePathForResourceName(NSString* name, NSString* extension) {
    NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
    if (file_path == NULL) {
        NSLog(@"Couldn't find '%@.%@' in bundle.", name, extension);
        exit(-1);
    }
    return file_path;
}

NSString* RunInferenceOnImage() {
    //NSString* graph = @"mobilenet_v1_1.0_224"; 
    NSString* graph = @"dog_retrained_mobilenet10_224_not_quantized";
    const int num_threads = 1;
    std::string input_layer_type = "float";
    std::vector<int> sizes = {1, 224, 224, 3};
    
    const NSString* graph_path = FilePathForResourceName(graph, @"tflite");
    
    std::unique_ptr<tflite::FlatBufferModel> model(tflite::FlatBufferModel::BuildFromFile([graph_path UTF8String]));
    if (!model) {
        NSLog(@"Failed to mmap model %@.", graph);
        exit(-1);
    }
    NSLog(@"Loaded model %@.", graph);
    model->error_reporter();
    NSLog(@"Resolved reporter.");

    tflite::ops::builtin::BuiltinOpResolver resolver;
    
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        NSLog(@"Failed to construct interpreter.");
        exit(-1);
    }
    
    if (num_threads != -1) {
        interpreter->SetNumThreads(num_threads);
    }
    
    int input = interpreter->inputs()[0];
    
    if (input_layer_type != "string") {
        interpreter->ResizeInputTensor(input, sizes);
    }
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        NSLog(@"Failed to allocate tensors.");
        exit(-1);
    }
    
    // Read the label list
    NSString* labels_path = FilePathForResourceName(@"labels", @"txt");
    std::vector<std::string> label_strings;
    std::ifstream t;
    t.open([labels_path UTF8String]);
    std::string line;
    while (t) {
        std::getline(t, line);
        label_strings.push_back(line);
    }
    t.close();
    
    NSString* image_path = FilePathForResourceName(@"lab1", @"jpg");
    int image_width;
    int image_height;
    int image_channels;
    std::vector<uint8_t> image_data = LoadImageFromFile([image_path UTF8String], &image_width, &image_height, &image_channels);
    const int wanted_width = 224;
    const int wanted_height = 224;
    const int wanted_channels = 3;
    const float input_mean = 127.5f;
    const float input_std = 127.5f;
    assert(image_channels >= wanted_channels);
    uint8_t* in = image_data.data();
    float* out = interpreter->typed_tensor<float>(input);
    for (int y = 0; y < wanted_height; ++y) {
        const int in_y = (y * image_height) / wanted_height;
        uint8_t* in_row = in + (in_y * image_width * image_channels);
        float* out_row = out + (y * wanted_width * wanted_channels);
        for (int x = 0; x < wanted_width; ++x) {
            const int in_x = (x * image_width) / wanted_width;
            uint8_t* in_pixel = in_row + (in_x * image_channels);
            float* out_pixel = out_row + (x * wanted_channels);
            for (int c = 0; c < wanted_channels; ++c) {
                out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
            }
        }
    }
    
    if (interpreter->Invoke() != kTfLiteOk) {
        NSLog(@"Failed to invoke!");
        exit(-1);
    }
    
    float* output = interpreter->typed_output_tensor<float>(0);
    const int output_size = 1000;
    const int kNumResults = 5;
    const float kThreshold = 0.1f;
    std::vector<std::pair<float, int> > top_results;
    GetTopN(output, output_size, kNumResults, kThreshold, &top_results);
    
    std::stringstream ss;
    ss.precision(3);
    for (const auto& result : top_results) {
        const float confidence = result.first;
        const int index = result.second;
        
        ss << index << " " << confidence << "  ";
        
        // Write out the result as a string
        if (index < label_strings.size()) {
            // just for safety: theoretically, the output is under 1000 unless there
            // is some numerical issues leading to a wrong prediction.
            ss << label_strings[index];
        } else {
            ss << "Prediction: " << index;
        }
        
        ss << "\n";
    }
    
    std::string predictions = ss.str();
    NSString* result = @"";
    result = [NSString stringWithFormat:@"%@ - %s", result, predictions.c_str()];
    NSLog(@"Predictions: %@", result);
    return result;
}

