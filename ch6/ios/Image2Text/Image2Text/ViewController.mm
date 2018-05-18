//
//  ViewController.m
//  Image2Text
//
//  Created by Jeff Tang on 2/3/18.
//  Copyright © 2018 Jeff Tang. All rights reserved.
//

#import "ViewController.h"

#include <fstream>
#include <queue>
#include "tensorflow/core/framework/op_kernel.h"
#include "ios_image_load.h"

#include "tensorflow_utils.h"


using namespace std;


static NSString* MODEL_FILE = @"image2text_frozen_transformed";
static NSString* MODEL_FILE_MEMMAPPED = @"image2text_frozen_transformed_memmapped";
static NSString* MODEL_FILE_TYPE = @"pb";
static NSString* VOCAB_FILE = @"word_counts";
static NSString* VOCAB_FILE_TYPE = @"txt";
static NSString *image_name = @"im2txt4.png";

const string INPUT_NODE1 = "convert_image/Cast";
const string OUTPUT_NODE1 = "lstm/initial_state";
const string INPUT_NODE2 = "input_feed";
const string INPUT_NODE3 = "lstm/state_feed";
const string OUTPUT_NODE2 = "softmax";
const string OUTPUT_NODE3 = "lstm/state";

const int wanted_width = 299;
const int wanted_height = 299;
const int wanted_channels = 3;

const int CAPTION_LEN = 20;
const int START_ID = 2;
const int END_ID = 3;
const int WORD_COUNT = 12000;
const int STATE_COUNT = 1024;

unique_ptr<tensorflow::Session> session;
unique_ptr<tensorflow::MemmappedEnv> tf_memmapped_env;

std::vector<std::string> words;

UIImageView *_iv;
UILabel *_lbl;

NSString* generateCaption(bool memmapped);


@interface ViewController ()

@end

@implementation ViewController

-(void)tapped:(UITapGestureRecognizer *)tapGestureRecognizer {
    
    UIAlertAction* memmapped = [UIAlertAction actionWithTitle:@"Use Memmapped Model" style:UIAlertActionStyleDefault handler:^(UIAlertAction * action) {
        _lbl.text = @"Processing...";
        _iv.image = [UIImage imageNamed:image_name];
        dispatch_async(dispatch_get_global_queue(0, 0), ^{
            NSString *caption = generateCaption(true);
            
            dispatch_async(dispatch_get_main_queue(), ^{
                _lbl.text = caption;
            });
        });
    }];
    UIAlertAction* non_memmapped = [UIAlertAction actionWithTitle:@"Use Non-memmapped Model" style:UIAlertActionStyleDefault handler:^(UIAlertAction * action) {
        _lbl.text = @"Processing...";
        _iv.image = [UIImage imageNamed:image_name];
        dispatch_async(dispatch_get_global_queue(0, 0), ^{
            NSString *caption = generateCaption(false);
            dispatch_async(dispatch_get_main_queue(), ^{
                _lbl.text = caption;
            });
        });
    }];
    
    UIAlertAction* none = [UIAlertAction actionWithTitle:@"None" style:UIAlertActionStyleDefault handler:^(UIAlertAction * action) {}];
    
    UIAlertController* alert = [UIAlertController alertControllerWithTitle:@"Image2Text Model Pick" message:nil preferredStyle:UIAlertControllerStyleAlert];
    [alert addAction:memmapped];
    [alert addAction:non_memmapped];
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
                                                                 constant:60];
    [self.view addConstraint:horizontal];
    [self.view addConstraint:vertical];
    
    _iv = [[UIImageView alloc] initWithFrame:self.view.frame];
    _iv.contentMode = UIViewContentModeScaleAspectFit;
    [self.view addSubview:_iv];
    _iv.image = [UIImage imageNamed:image_name];
    
    UITapGestureRecognizer *recognizer = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(tapped:)];
    [self.view addGestureRecognizer:recognizer];
    
    
    // load the voc file once and for all
    NSString* voc_file_path = FilePathForResourceName(VOCAB_FILE, VOCAB_FILE_TYPE);
    if (!voc_file_path) {
        LOG(FATAL) << "Couldn't load vocabuary file: " << voc_file_path;
    }
    ifstream t;
    t.open([voc_file_path UTF8String]);
    string line;
    while(t){
        getline(t, line);
        size_t pos = line.find(" ");
        words.push_back(line.substr(0, pos));
    }
    t.close();
    
}



- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end


NSString* generateCaption(bool memmapped) {
    tensorflow::Status load_status;
    
    if (memmapped) {
        load_status = LoadMemoryMappedModel(MODEL_FILE_MEMMAPPED, MODEL_FILE_TYPE, &session, &tf_memmapped_env);
    }
    else {
        load_status = LoadModel(MODEL_FILE, MODEL_FILE_TYPE, &session);
    }
    
    if (!load_status.ok()) {
        LOG(FATAL) << "Couldn't load model: " << load_status;
        return @"Couldn't load model";
    }

    int image_width;
    int image_height;
    int image_channels;
    NSArray *name_ext = [image_name componentsSeparatedByString:@"."];
    NSString* image_path = FilePathForResourceName(name_ext[0], name_ext[1]);
    std::vector<tensorflow::uint8> image_data = LoadImageFromFile([image_path UTF8String], &image_width, &image_height, &image_channels);
    
    tensorflow::Tensor image_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({wanted_height, wanted_width, wanted_channels}));
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
                out_pixel[c] = in_pixel[c];
            }
        }
    }
    
    
    vector<tensorflow::Tensor> initial_state;
    
    if (session.get()) {
        tensorflow::Status run_status = session->Run({{INPUT_NODE1, image_tensor}}, {OUTPUT_NODE1}, {}, &initial_state);
        if (!run_status.ok()) {
            LOG(ERROR) << "Getting initial state failed:" << run_status;
            return @"Getting initial state failed";
        }
    }
    
    
    tensorflow::Tensor input_feed(tensorflow::DT_INT64, tensorflow::TensorShape({1,}));
    tensorflow::Tensor state_feed(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,STATE_COUNT}));
    
    auto input_feed_map = input_feed.tensor<int64_t, 1>();
    auto state_feed_map = state_feed.tensor<float, 2>();
    
    
    input_feed_map(0) = START_ID;
    
    auto initial_state_map = initial_state[0].tensor<float, 2>();
    for (int i = 0; i < STATE_COUNT; i++){
        state_feed_map(0,i) = initial_state_map(0,i);
    }

    vector<int> captions;
    for (int i=0; i<CAPTION_LEN; i++) {
        std::vector<tensorflow::Tensor> output;
        tensorflow::Status run_status = session->Run({{INPUT_NODE2, input_feed}, {INPUT_NODE3, state_feed}}, {OUTPUT_NODE2, OUTPUT_NODE3}, {}, &output);
        if (!run_status.ok()) {
            LOG(ERROR) << "Running model failed:" << run_status;
            return @"Getting lstm state failed";
        }
        else {
            tensorflow::Tensor softmax = output[0];
            tensorflow::Tensor state = output[1];
            
            auto softmax_map = softmax.tensor<float, 2>();
            auto state_map = state.tensor<float, 2>();
            
            // get the word id with the max prob in softmax
            float max_prob = 0.0f;
            int max_word_id = 0;
            for (int j = 0; j < WORD_COUNT; j++){
                if (softmax_map(0,j) > max_prob) {
                    max_prob = softmax_map(0,j);
                    max_word_id = j;
                }
            }
            
            if (max_word_id == END_ID) break;
            
            captions.push_back(max_word_id);
            
            input_feed_map(0) = max_word_id;
            for (int j = 0; j < STATE_COUNT; j++){
                state_feed_map(0,j) = state_map(0,j);
            }
            
        }
    }
    
    NSString *sentence = @"";
    for (int i=0; i<captions.size(); i++) {
        if (captions[i] == START_ID) continue;
        if (captions[i] == END_ID) break;
        
        sentence = [NSString stringWithFormat:@"%@ %s", sentence, words[captions[i]].c_str()];
        
    }
    
    
    
    NSLog(@"Caption: %@", sentence);
    
    
    return sentence;
    
}

