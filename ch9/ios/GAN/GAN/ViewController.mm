//
//  ViewController.m
//  GAN
//
//  Created by Jeff Tang on 3/2/18.
//  Copyright © 2018 Jeff Tang. All rights reserved.
//

#import "ViewController.h"

#include <fstream>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

#include <queue>
#include <fstream>
#include <random>
#include "tensorflow_utils.h"

#include "ios_image_load.h"

using namespace std;

unique_ptr<tensorflow::Session> tf_session;
unique_ptr<tensorflow::MemmappedEnv> tf_memmapped_env;

UITextView *_tv;
UIButton *_btn;
int wanted_width = 256;
int wanted_height = 256;
static NSString *image_name = @"ww.png";
UIImageView *_iv;

@interface ViewController ()

@end

@implementation ViewController

- (IBAction)btnTapped:(id)sender {
    UIAlertAction* mnist = [UIAlertAction actionWithTitle:@"Generate Digits" style:UIAlertActionStyleDefault handler:^(UIAlertAction * action) {
        _iv.image = NULL;
        dispatch_async(dispatch_get_global_queue(0, 0), ^{
            NSArray *arrayGreyscaleValues = [self runMNISTModel];
            dispatch_async(dispatch_get_main_queue(), ^{
                UIImage *imgDigit = [self createMNISTImageInRect:_iv.frame values:arrayGreyscaleValues];
                _iv.image = imgDigit;
            });
        });

        
    }];
    UIAlertAction* pix2pix = [UIAlertAction actionWithTitle:@"Enhance Image" style:UIAlertActionStyleDefault handler:^(UIAlertAction * action) {
        _iv.image = [UIImage imageNamed:image_name];
        dispatch_async(dispatch_get_global_queue(0, 0), ^{
            NSArray *arrayRGBValues = [self runPix2PixBlurryModel];
            dispatch_async(dispatch_get_main_queue(), ^{
                UIImage *imgTranslated = [self createTranslatedImageInRect:_iv.frame values:arrayRGBValues];
                _iv.image = imgTranslated;
            });
        });
    }];
    
    UIAlertAction* none = [UIAlertAction actionWithTitle:@"None" style:UIAlertActionStyleDefault handler:^(UIAlertAction * action) {}];
    
    UIAlertController* alert = [UIAlertController alertControllerWithTitle:@"Use GAN to" message:nil preferredStyle:UIAlertControllerStyleAlert];
    [alert addAction:mnist];
    [alert addAction:pix2pix];
    [alert addAction:none];
    [self presentViewController:alert animated:YES completion:nil];
}




- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    _iv = [[UIImageView alloc] initWithFrame:self.view.frame];
    _iv.contentMode = UIViewContentModeScaleAspectFit;
    [self.view addSubview:_iv];
    _iv.image = [UIImage imageNamed:image_name];
    
    
    _btn = [UIButton buttonWithType:UIButtonTypeSystem];
    [_btn setTranslatesAutoresizingMaskIntoConstraints:NO];
    _btn.titleLabel.font = [UIFont systemFontOfSize:32];
    [_btn setTitle:@"GAN" forState:UIControlStateNormal];
    [self.view addSubview:_btn];
    
    NSLayoutConstraint *horizontal = [NSLayoutConstraint constraintWithItem:_btn
                                                                  attribute:NSLayoutAttributeCenterX
                                                                  relatedBy:NSLayoutRelationEqual
                                                                     toItem:self.view
                                                                  attribute:NSLayoutAttributeCenterX
                                                                 multiplier:1
                                                                   constant:0];
    NSLayoutConstraint *vertical = [NSLayoutConstraint constraintWithItem:_btn
                                                                attribute:NSLayoutAttributeCenterY
                                                                relatedBy:NSLayoutRelationEqual
                                                                   toItem:self.view
                                                                attribute:NSLayoutAttributeBottom
                                                               multiplier:1
                                                                 constant:-50];
    [self.view addConstraint:horizontal];
    [self.view addConstraint:vertical];
    
    [_btn addTarget:self action:@selector(btnTapped:) forControlEvents:UIControlEventTouchUpInside];
    
}



- (NSArray*) runMNISTModel {
    tensorflow::Status load_status;
    
    load_status = LoadModel(@"gan_mnist", @"pb", &tf_session);
    if (!load_status.ok()) return NULL;
    
    
    std::string input_layer = "z_placeholder";
    std::string output_layer = "Sigmoid_1";
 
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({6, 100}));
    auto input_map = input_tensor.tensor<float, 2>();
    
    unsigned seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    
    std::normal_distribution<double> distribution(0.0, 1.0);
    
    for (int i = 0; i < 6; i++){
        for (int j = 0; j < 100; j++){
            double number = distribution(generator);
            //NSLog(@"%f\n", number);
            input_map(i,j) = number;
        }
    }
    
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::Status run_status = tf_session->Run({{input_layer, input_tensor}},
                                                 {output_layer}, {}, &outputs);
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        return NULL;
    }
    tensorflow::string status_string = run_status.ToString();
    tensorflow::Tensor* output_tensor = &outputs[0];
    
    
    const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>, Eigen::Aligned>& output = output_tensor->flat<float>();
    const long count = output.size();
    NSMutableArray *arrayGreyscaleValues = [NSMutableArray array];
    
    for (int i = 0; i < count; ++i) {
        //printf("\n%f", output(i));
        const float value = output(i);
        [arrayGreyscaleValues addObject:[NSNumber numberWithFloat:value]];
    }
    
    return arrayGreyscaleValues;

}




- (NSArray*) runPix2PixBlurryModel {
    tensorflow::Status load_status;
    
    load_status = LoadMemoryMappedModel(@"pix2pix_transformed_memmapped", @"pb", &tf_session, &tf_memmapped_env);
    if (!load_status.ok()) return NULL;
    
    std::string input_layer = "image_feed";
    std::string output_layer = "generator_1/deprocess/truediv";
    

    NSString* image_path = FilePathForResourceName(@"ww", @"png");
    int image_width;
    int image_height;
    int image_channels;
    std::vector<tensorflow::uint8> image_data = LoadImageFromFile([image_path UTF8String], &image_width, &image_height, &image_channels);
    
    const int wanted_channels = 3;
    const float input_mean = 128.0f;
    const float input_std = 128.0f;
    
    assert(image_channels >= wanted_channels);
    tensorflow::Tensor image_tensor(
                                    tensorflow::DT_FLOAT,
                                    tensorflow::TensorShape({
        1, wanted_height, wanted_width, wanted_channels}));
    auto image_tensor_mapped = image_tensor.tensor<float, 4>();
    tensorflow::uint8* in = image_data.data();
    // tensorflow::uint8* in_end = (in + (image_height * image_width * image_channels));
    float* out = image_tensor_mapped.data();
    for (int y = 0; y < wanted_height; ++y) {
        const int in_y = (y * image_height) / wanted_height;
        tensorflow::uint8* in_row = in + (in_y * image_width * image_channels);
        float* out_row = out + (y * wanted_width * wanted_channels);
        for (int x = 0; x < wanted_width; ++x) {
            const int in_x = (x * image_width) / wanted_width;
            tensorflow::uint8* in_pixel = in_row + (in_x * image_channels);
            float* out_pixel = out_row + (x * wanted_channels);
            for (int c = 0; c < wanted_channels; ++c) {
                out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
                //if (x<2) printf("%f\n", out_pixel[c]);
            }
        }
    }

    
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::Status run_status = tf_session->Run({{input_layer, image_tensor}},
                                                    {output_layer}, {}, &outputs);
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        return NULL;
    }
    tensorflow::string status_string = run_status.ToString();
    tensorflow::Tensor* output_tensor = &outputs[0];
    
    const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>, Eigen::Aligned>& output = output_tensor->flat<float>();

    const long count = output.size(); // 256*256*3
    NSMutableArray *arrayRGBValues = [NSMutableArray array];
    
    for (int i = 0; i < count; ++i) {
        const float value = output(i);
        [arrayRGBValues addObject:[NSNumber numberWithFloat:value]];
    }
    return arrayRGBValues;
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


- (UIImage *)createMNISTImageInRect:(CGRect)rect values:(NSArray*)greyscaleValues
{
    UIGraphicsBeginImageContextWithOptions(CGSizeMake(rect.size.width, rect.size.height), NO, 0.0);
    int i=0;
    const int size = 3;
    NSLog(@"greyscaleValues count=%ld", [greyscaleValues count]);
    for (NSNumber *val in greyscaleValues) {
        float c = [val floatValue];
        int x = i%28;
        int y = i/28;
        i++;

        CGRect rect = CGRectMake(145+size*x, 50+y*size, size, size);
        UIBezierPath *path = [UIBezierPath bezierPathWithRect:rect];
        UIColor *color = [UIColor colorWithRed:c green:c blue:c alpha:1.0];
        [color setFill];
        [path fill];
    }
    
    UIImage *image = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return image;
}


- (UIImage *)createTranslatedImageInRect:(CGRect)rect values:(NSArray*)rgbValues
{
    UIGraphicsBeginImageContextWithOptions(CGSizeMake(wanted_width, wanted_height), NO, 0.0);
    NSLog(@"rgbValues count=%ld", [rgbValues count]);
    for (int i=0; i<256*256; i++) {
        float R = [rgbValues[i*3] floatValue];
        float G = [rgbValues[i*3+1] floatValue];
        float B = [rgbValues[i*3+2] floatValue];
        const int size = 1;
        int x = i%256;
        int y = i/256;
        CGRect rect = CGRectMake(size*x, y*size, size, size);
        UIBezierPath *path = [UIBezierPath bezierPathWithRect:rect];
        UIColor *color = [UIColor colorWithRed:R green:G blue:B alpha:1.0];
        
        [color setFill];
        [path fill];
    }
    
    UIImage *image = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return image;
}

@end
