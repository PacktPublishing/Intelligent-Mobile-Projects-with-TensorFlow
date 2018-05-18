//
//  ViewController.m
//  NeuralStyleTransfer
//
//  Created by Jeff Tang on 1/16/18.
//  Copyright © 2018 Jeff Tang. All rights reserved.
//

#import "ViewController.h"
#include <fstream>
#include <queue>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

#include "ios_image_load.h"

const int NUM_STYLES = 26;
const int wanted_width = 420; //420; //300;
const int wanted_height = 560; // 560; //400;
static NSString *image_name = @"ww.jpg";
UIImageView *_iv;
UILabel *_lbl;

UIImage* imageStyleTransfer(NSString *model);

namespace {
    class IfstreamInputStream : public ::google::protobuf::io::CopyingInputStream {
    public:
        explicit IfstreamInputStream(const std::string& file_name)
        : ifs_(file_name.c_str(), std::ios::in | std::ios::binary) {}
        ~IfstreamInputStream() { ifs_.close(); }
        
        int Read(void* buffer, int size) {
            if (!ifs_) {
                return -1;
            }
            ifs_.read(static_cast<char*>(buffer), size);
            return ifs_.gcount();
        }
        
    private:
        std::ifstream ifs_;
    };
}  // namespace


@interface UIImage (Resize)
- (UIImage*)scaleToSize:(CGSize)size;
@end


@implementation UIImage (Resizing)

- (UIImage*)scaleToSize:(CGSize)size {
    UIGraphicsBeginImageContext(size);
    
    CGContextRef context = UIGraphicsGetCurrentContext();
    CGContextTranslateCTM(context, 0.0, size.height);
    CGContextScaleCTM(context, 1.0, -1.0);
    
    CGContextDrawImage(context, CGRectMake(0.0f, 0.0f, size.width, size.height), self.CGImage);
    
    UIImage* scaledImage = UIGraphicsGetImageFromCurrentImageContext();
    
    UIGraphicsEndImageContext();
    
    return scaledImage;
}
@end

@interface ViewController ()

@end

@implementation ViewController
-(void)tapped:(UITapGestureRecognizer *)tapGestureRecognizer {
    
    UIAlertAction* fast_style_transfer = [UIAlertAction actionWithTitle:@"Fast Style Transfer" style:UIAlertActionStyleDefault handler:^(UIAlertAction * action) {
        _lbl.text = @"Processing...";
        _iv.image = [UIImage imageNamed:image_name];
        dispatch_async(dispatch_get_global_queue(0, 0), ^{
            //UIImage *img = imageStyleTransfer(@"fst_frozen_quantized");
            UIImage *img = imageStyleTransfer(@"fnn_starry_night_420x560");
            dispatch_async(dispatch_get_main_queue(), ^{
                _lbl.text = @"Tap Anywhere";
                _iv.image = img;
            });
        });
    }];
    UIAlertAction* multi_style_transfer = [UIAlertAction actionWithTitle:@"Multistyle Transfer" style:UIAlertActionStyleDefault handler:^(UIAlertAction * action) {
        _lbl.text = @"Processing...";
        _iv.image = [UIImage imageNamed:image_name];
        dispatch_async(dispatch_get_global_queue(0, 0), ^{
            UIImage *img = imageStyleTransfer(@"stylize_quantized");
            dispatch_async(dispatch_get_main_queue(), ^{
                _lbl.text = @"Tap Anywhere";
                _iv.image = img;
            });
        });
    }];

    UIAlertAction* none = [UIAlertAction actionWithTitle:@"None" style:UIAlertActionStyleDefault handler:^(UIAlertAction * action) {}];
    
    UIAlertController* alert = [UIAlertController alertControllerWithTitle:@"Pick a Model" message:nil preferredStyle:UIAlertControllerStyleAlert];
    [alert addAction:fast_style_transfer];
    [alert addAction:multi_style_transfer];
    [alert addAction:none];
    [self presentViewController:alert animated:YES completion:nil];
    
}

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    _lbl = [[UILabel alloc] init];
    [_lbl setTranslatesAutoresizingMaskIntoConstraints:NO];
    _lbl.text = @"Tap Anywhere";
    [self.view addSubview:_lbl];
    
    NSLayoutConstraint *horizontal = [NSLayoutConstraint constraintWithItem:_lbl
                                                                  attribute:NSLayoutAttributeCenterX
                                                                  relatedBy:NSLayoutRelationEqual
                                                                     toItem:self.view
                                                                  attribute:NSLayoutAttributeCenterX
                                                                 multiplier:1
                                                                   constant:0];
    NSLayoutConstraint *vertical = [NSLayoutConstraint constraintWithItem:_lbl
                                                                attribute:NSLayoutAttributeTop
                                                                relatedBy:NSLayoutRelationEqual
                                                                   toItem:self.view
                                                                attribute:NSLayoutAttributeTop
                                                               multiplier:1
                                                                 constant:40];
    [self.view addConstraint:horizontal];
    [self.view addConstraint:vertical];
    
    _iv = [[UIImageView alloc] initWithFrame:self.view.frame];
    _iv.contentMode = UIViewContentModeScaleAspectFit;
    [self.view addSubview:_iv];
    _iv.image = [UIImage imageNamed:image_name];
    
    UITapGestureRecognizer *recognizer = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(tapped:)];
    [self.view addGestureRecognizer:recognizer];

}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


+ (UIImage *) convertRGBBufferToUIImage:(unsigned char *) buffer
                                withWidth:(int) width
                               withHeight:(int) height {
    
    // added code
    char* rgba = (char*)malloc(width*height*4);
    for(int i=0; i < width*height; ++i) {
        rgba[4*i] = buffer[3*i];
        rgba[4*i+1] = buffer[3*i+1];
        rgba[4*i+2] = buffer[3*i+2];
        rgba[4*i+3] = 255;
    }
    //
    
    
    size_t bufferLength = width * height * 4;
    CGDataProviderRef provider = CGDataProviderCreateWithData(NULL, rgba, bufferLength, NULL);
    size_t bitsPerComponent = 8;
    size_t bitsPerPixel = 32;
    size_t bytesPerRow = 4 * width;
    
    CGColorSpaceRef colorSpaceRef = CGColorSpaceCreateDeviceRGB();
    if(colorSpaceRef == NULL) {
        NSLog(@"Error allocating color space");
        CGDataProviderRelease(provider);
        return nil;
    }
    
    CGBitmapInfo bitmapInfo = kCGBitmapByteOrderDefault | kCGImageAlphaPremultipliedLast;
    CGColorRenderingIntent renderingIntent = kCGRenderingIntentDefault;
    
    CGImageRef iref = CGImageCreate(width,
                                    height,
                                    bitsPerComponent,
                                    bitsPerPixel,
                                    bytesPerRow,
                                    colorSpaceRef,
                                    bitmapInfo,
                                    provider,   // data provider
                                    NULL,       // decode
                                    YES,            // should interpolate
                                    renderingIntent);
    
    uint32_t* pixels = (uint32_t*)malloc(bufferLength);
    
    if(pixels == NULL) {
        NSLog(@"Error: Memory not allocated for bitmap");
        CGDataProviderRelease(provider);
        CGColorSpaceRelease(colorSpaceRef);
        CGImageRelease(iref);
        return nil;
    }
    
    CGContextRef context = CGBitmapContextCreate(pixels,
                                                 width,
                                                 height,
                                                 bitsPerComponent,
                                                 bytesPerRow,
                                                 colorSpaceRef,
                                                 bitmapInfo);
    
    if(context == NULL) {
        NSLog(@"Error context not created");
        free(pixels);
    }
    
    UIImage *image = nil;
    if(context) {
        
        CGContextDrawImage(context, CGRectMake(0.0f, 0.0f, width, height), iref);
        
        CGImageRef imageRef = CGBitmapContextCreateImage(context);
        
        // Support both iPad 3.2 and iPhone 4 Retina displays with the correct scale
        if([UIImage respondsToSelector:@selector(imageWithCGImage:scale:orientation:)]) {
            float scale = [[UIScreen mainScreen] scale];
            image = [UIImage imageWithCGImage:imageRef scale:scale orientation:UIImageOrientationUp];
        } else {
            image = [UIImage imageWithCGImage:imageRef];
        }
        
        CGImageRelease(imageRef);
        CGContextRelease(context);
    }
    
    CGColorSpaceRelease(colorSpaceRef);
    CGImageRelease(iref);
    CGDataProviderRelease(provider);
    
    if(pixels) {
        free(pixels);
    }
    return image;
}

@end


static UIImage* tensorToUIImage(NSString *model,
                               const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
                               Eigen::Aligned>& outputTensor, int image_width, int image_height) {
    const int count = outputTensor.size();
    unsigned char* buffer = (unsigned char*)malloc(count);

    for (int i = 0; i < count; ++i) {
        // 255 * outputTensor(i) is the outputTensor value is in the range of 0-1, 1.0 if the range is 0-255
        const float value = ([model isEqualToString:@"stylize_quantized"] ? 255.0 : 1.0) * outputTensor(i);
        int n;
        if (value < 0) n = 0;
        else if (value > 255) n = 255;
        else n = (int)value;
        buffer[i] = n;
    }
    
    UIImage *img = [ViewController convertRGBBufferToUIImage:buffer withWidth:wanted_width withHeight:wanted_height];
    UIImage *imgScaled = [img scaleToSize:CGSizeMake(image_width, image_height)];
    
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *filePath = [[paths objectAtIndex:0] stringByAppendingPathComponent:@"fns.jpg"];
    
    // Save image.
    [UIImageJPEGRepresentation(imgScaled, 1.0) writeToFile:filePath atomically:YES];
    
    NSLog(@"filePath=%@", filePath);
    
    return imgScaled;
    
}


bool PortableReadFileToProto(const std::string& file_name,
                             ::google::protobuf::MessageLite* proto) {
    ::google::protobuf::io::CopyingInputStreamAdaptor stream(
                                                             new IfstreamInputStream(file_name));
    stream.SetOwnsCopyingStream(true);
    // TODO(jiayq): the following coded stream is for debugging purposes to allow
    // one to parse arbitrarily large messages for MessageLite. One most likely
    // doesn't want to put protobufs larger than 64MB on Android, so we should
    // eventually remove this and quit loud when a large protobuf is passed in.
    ::google::protobuf::io::CodedInputStream coded_stream(&stream);
    // Total bytes hard limit / warning limit are set to 1GB and 512MB
    // respectively.
    coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
    return proto->ParseFromCodedStream(&coded_stream);
}

NSString* FilePathForResourceName(NSString* name, NSString* extension) {
    NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
    if (file_path == NULL) {
        LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "."
        << [extension UTF8String] << "' in bundle.";
    }
    return file_path;
}


UIImage* imageStyleTransfer(NSString *model) {
    tensorflow::SessionOptions options;
    
    tensorflow::Session* session_pointer = nullptr;
    tensorflow::Status session_status = tensorflow::NewSession(options, &session_pointer);
    if (!session_status.ok()) {
        std::string status_string = session_status.ToString();
        return NULL; //[NSString stringWithFormat: @"Session create failed - %s",status_string.c_str()];
    }
    std::unique_ptr<tensorflow::Session> session(session_pointer);
    LOG(INFO) << "Session created.";
    
    tensorflow::GraphDef tensorflow_graph;
    LOG(INFO) << "Graph created.";
    
    //NSString* network_path = FilePathForResourceName(@"fst_frozen_quantized", @"pb");
    //NSString* network_path = FilePathForResourceName(@"quantized_image_stylization_frozen_graph", @"pb");
    NSString* network_path = FilePathForResourceName(model, @"pb");
    
    PortableReadFileToProto([network_path UTF8String], &tensorflow_graph);
    
    LOG(INFO) << "Creating session.";
    tensorflow::Status s = session->Create(tensorflow_graph);
    if (!s.ok()) {
        LOG(ERROR) << "Could not create TensorFlow Graph: " << s;
        return NULL; //@"";
    }
    
    NSArray *name_ext = [image_name componentsSeparatedByString:@"."];
    NSString* image_path = FilePathForResourceName(name_ext[0], name_ext[1]);

    int image_width;
    int image_height;
    int image_channels;
    std::vector<tensorflow::uint8> image_data = LoadImageFromFile(
                                                                  [image_path UTF8String], &image_width, &image_height, &image_channels);
    
    const int wanted_channels = 3;
    const float input_mean = 128.0f;
    const float input_std = 128.0f;
    
    assert(image_channels >= wanted_channels);

    
    if (![model isEqualToString:@"stylize_quantized"]) {
        tensorflow::Tensor image_tensor(
                                        tensorflow::DT_FLOAT,
                                        tensorflow::TensorShape({
            wanted_height, wanted_width, wanted_channels}));
        auto image_tensor_mapped = image_tensor.tensor<float, 3>();
        tensorflow::uint8* in = image_data.data();
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
                }
            }
        }
        
        std::string input_layer = "img_placeholder";
        std::string output_layer = "preds";
        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status run_status = session->Run({{input_layer, image_tensor}},
                                                     {output_layer}, {}, &outputs);
        if (!run_status.ok()) {
            LOG(ERROR) << "Running model failed: " << run_status;
            return NULL;
        }
        tensorflow::string status_string = run_status.ToString();
        tensorflow::Tensor* output = &outputs[0];
        
        UIImage *imgScaled = tensorToUIImage(model, output->flat<float>(), image_width, image_height);
        return imgScaled;
    }
    else {
        std::string input_layer = "input";
        std::string style_layer = "style_num";
        std::string output_layer = "transformer/expand/conv3/conv/Sigmoid";
        
        tensorflow::Tensor image_tensor(
                                        tensorflow::DT_FLOAT,
                                        tensorflow::TensorShape({
            1, wanted_height, wanted_width, wanted_channels}));
        auto image_tensor_mapped = image_tensor.tensor<float, 4>();
        tensorflow::uint8* in = image_data.data();
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
                }
            }
        }
        
        tensorflow::Tensor style_tensor(tensorflow::DT_FLOAT,
                                        tensorflow::TensorShape({ NUM_STYLES, 1}));
        auto style_tensor_mapped = style_tensor.tensor<float, 2>();
        float* out_style = style_tensor_mapped.data();
        for (int i = 0; i < NUM_STYLES; i++) {
            out_style[i] = 1.0 / NUM_STYLES; // 0.0
        }
//        out_style[19] = 1.0;

//        out_style[4] = 0.5;
//        out_style[19] = 0.5;

//        auto stensor = style_tensor.matrix<float>();
//        stensor.setConstant(1.0f / NUM_STYLES);

        
        std::vector<tensorflow::Tensor> outputs;
        tensorflow::Status run_status = session->Run({{input_layer, image_tensor}, {style_layer, style_tensor}},
                                                     {output_layer}, {}, &outputs);
        if (!run_status.ok()) {
            LOG(ERROR) << "Running model failed: " << run_status;
            return NULL;
        }
        tensorflow::string status_string = run_status.ToString();
        tensorflow::Tensor* output = &outputs[0];
        
        UIImage *imgScaled = tensorToUIImage(model, output->flat<float>(), image_width, image_height);
        return imgScaled;
    }
}



