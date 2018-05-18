//
//  ViewController.m
//  StockPrice
//
//  Created by Jeff Tang on 2/20/18.
//  Copyright © 2018 Jeff Tang. All rights reserved.
//

#import "ViewController.h"

#include <fstream>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

#include <queue>
#include <fstream>
#include "tensorflow_utils.h"
using namespace std;

unique_ptr<tensorflow::Session> tf_session;

UITextView *_tv;
UIButton *_btn;
NSMutableArray *_closeprices;
const int SEQ_LEN = 20;


@interface ViewController ()

@end

@implementation ViewController
- (IBAction)btnTapped:(id)sender {
    NSLog(@">>>%@", [NSThread currentThread]);

    
    UIAlertAction* tf = [UIAlertAction actionWithTitle:@"Use TensorFlow Model" style:UIAlertActionStyleDefault handler:^(UIAlertAction * action) {
        [self getLatestData:NO];
    }];
    UIAlertAction* keras = [UIAlertAction actionWithTitle:@"Use Keras Model" style:UIAlertActionStyleDefault handler:^(UIAlertAction * action) {
        [self getLatestData:YES];
    }];
    
    UIAlertAction* none = [UIAlertAction actionWithTitle:@"None" style:UIAlertActionStyleDefault handler:^(UIAlertAction * action) {}];
    
    UIAlertController* alert = [UIAlertController alertControllerWithTitle:@"RNN Model Pick" message:nil preferredStyle:UIAlertControllerStyleAlert];
    [alert addAction:tf];
    [alert addAction:keras];
    [alert addAction:none];
    [self presentViewController:alert animated:YES completion:nil];
}


-(void)getLatestData:(BOOL)useKerasModel {
    
    NSURLSession *session = [NSURLSession sharedSession];
    [[session dataTaskWithURL:[NSURL URLWithString:@"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=amzn&apikey=4SOSJM2XCRIB5IUS&datatype=csv&outputsize=compact"]
            completionHandler:^(NSData *data,
                                NSURLResponse *response,
                                NSError *error) {
                NSString *stockinfo = [[NSString alloc] initWithData:data encoding:NSASCIIStringEncoding];
                // timestamp,open,high,low,close,volume
                // 2018-02-23,1495.3400,1500.0000,1486.5000,1500.0000,4327008
                // 2018-02-22,1495.3600,1502.5400,1475.7600,1485.3400,4732555
                // 2018-02-21,1485.0000,1503.4900,1478.9200,1482.9200,6216694
                NSArray *lines = [stockinfo componentsSeparatedByString:@"\n"];
                _closeprices = [NSMutableArray array];
                for (int i=0; i<SEQ_LEN; i++) {
                    NSArray *items = [lines[i+1] componentsSeparatedByString:@","];
                    [_closeprices addObject:items[4]];
                }
                
                
                for (NSString *p in _closeprices) {
                    NSLog(@"***** %@", p);
                }
                
                
                if (useKerasModel)
                    [self runKerasModel];
                else
                    [self runTFModel];
                
            }] resume];
    
}


- (void) runTFModel {
    NSLog(@"%@", [NSThread currentThread]);
    tensorflow::Status load_status;
    
    // no normalization is used
    load_status = LoadModel(@"amzn_tf_frozen", @"pb", &tf_session);
    
    
    tensorflow::Tensor prices(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, SEQ_LEN, 1}));
    
    auto prices_map = prices.tensor<float, 3>();

//    float p[] = {761.01,
//        761.09,
//        769.69,
//        778.52,
//        775.1 ,
//        780.22,
//        789.74,
//        804.7 ,
//        805.75,
//        799.16,
//        816.11,
//        828.72,
//        829.05,
//        837.31,
//        836.74,
//        834.03,
//        844.36,
//        841.66,
//        839.43,
//        841.71};
//    for (int i = 0; i < SEQ_LEN; i++){
//        prices_map(0,i,0) = p[i];
//    }

    NSString *txt = @"Last 20 Days:\n";

    for (int i = 0; i < SEQ_LEN; i++){
        prices_map(0,i,0) = [_closeprices[SEQ_LEN-i-1] floatValue];
        NSLog(@"%f", [_closeprices[SEQ_LEN-i-1] floatValue]);
        txt = [NSString stringWithFormat:@"%@%@\n", txt, _closeprices[SEQ_LEN-i-1]];
    }
    
    NSLog(@"*********");

    std::vector<tensorflow::Tensor> output;
    tensorflow::Status run_status = tf_session->Run({{"Placeholder", prices}}, {"preds"}, {}, &output);
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed:" << run_status;
    }
    else {
        tensorflow::Tensor preds = output[0];
        
        auto preds_map = preds.tensor<float, 2>();
        
        for (int j = 0; j < SEQ_LEN; j++){ // x1, x2, ..., x20 predicts x2, x3, ..., x21
            NSLog(@"%f", preds_map(0,j)); // the last one is the prediction we look for.
            /* exactly match the output of python predict.py !!!
             2018-02-24 21:10:08.362 StockPrice[5337:5583221] 758.096558
             2018-02-24 21:10:08.362 StockPrice[5337:5583221] 760.644958
             2018-02-24 21:10:08.363 StockPrice[5337:5583221] 769.116821
             2018-02-24 21:10:08.363 StockPrice[5337:5583221] 780.055908
             2018-02-24 21:10:08.363 StockPrice[5337:5583221] 775.078186
             2018-02-24 21:10:08.363 StockPrice[5337:5583221] 780.472595
             2018-02-24 21:10:08.364 StockPrice[5337:5583221] 790.768311
             2018-02-24 21:10:08.364 StockPrice[5337:5583221] 806.190247
             2018-02-24 21:10:08.364 StockPrice[5337:5583221] 806.636047
             2018-02-24 21:10:08.364 StockPrice[5337:5583221] 798.925598
             2018-02-24 21:10:08.365 StockPrice[5337:5583221] 817.802917
             2018-02-24 21:10:08.365 StockPrice[5337:5583221] 830.971985
             2018-02-24 21:10:08.366 StockPrice[5337:5583221] 828.890930
             2018-02-24 21:10:08.366 StockPrice[5337:5583221] 838.344727
             2018-02-24 21:10:08.366 StockPrice[5337:5583221] 838.014099
             2018-02-24 21:10:08.367 StockPrice[5337:5583221] 834.623901
             2018-02-24 21:10:08.367 StockPrice[5337:5583221] 845.585754
             2018-02-24 21:10:08.368 StockPrice[5337:5583221] 842.912170
             2018-02-24 21:10:08.368 StockPrice[5337:5583221] 840.306885
             2018-02-24 21:10:08.368 StockPrice[5337:5583221] 842.973450
             */
        }
        
        txt = [NSString stringWithFormat:@"%@\nPrediction with TF RNN model:\n%f", txt, preds_map(0,SEQ_LEN-1)];
        dispatch_async(dispatch_get_main_queue(), ^{
            [_tv setText:txt];
            [_tv sizeToFit];
        });
        
        NSLog(@"Prediction: %f", preds_map(0,SEQ_LEN-1));

    }
    
}


- (void) runKerasModel {
    tensorflow::Status load_status;
    
    // normalizatiokn is used
    load_status = LoadModel(@"amzn_keras_frozen", @"pb", &tf_session);
    if (!load_status.ok()) return;
    
    
    tensorflow::Tensor prices(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, SEQ_LEN, 1}));
    
    auto prices_map = prices.tensor<float, 3>();
    float lower = 5.97;
    float scale = 1479.37;
//    float p[] = {
//        0.40294855,
//        0.39574954,
//        0.39789235,
//        0.39879138,
//        0.40368535,
//        0.41156033,
//        0.41556879,
//        0.41904324,
//        0.42543786,
//        0.42040193,
//        0.42384258,
//        0.42249741,
//        0.4153998 ,
//        0.41925279,
//        0.41295281,
//        0.40598363,
//        0.40289448,
//        0.44182321,
//        0.45822208,
//        0.44975226};
    NSString *txt = @"Last 20 Days:\n";
    for (int i = 0; i < SEQ_LEN; i++){
        prices_map(0,i,0) = ([_closeprices[SEQ_LEN-i-1] floatValue] - lower)/scale;
//        prices_map(0,i,0) = p[i];
        NSLog(@">>> %f = %f", prices_map(0,i,0), scale * prices_map(0,i,0) + lower);
        txt = [NSString stringWithFormat:@"%@%@\n", txt, _closeprices[SEQ_LEN-i-1]];
    }

    
    // model output node name: activation_1/Identity
    // bidirectional_1_input if model is trained with model.add(Bidirectional(LSTM(units=100, return_sequences=True, input_shape=(None, 1)), input_shape=(seq_len, 1)))
    // lstm_1_input if model is trained with model.add(LSTM(units=50, return_sequences=True, input_shape=(None, 1)))
    
    std::vector<tensorflow::Tensor> output;
    tensorflow::Status run_status = tf_session->Run({{"bidirectional_1_input", prices}}, {"activation_1/Identity"}, {}, &output);
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed:" << run_status;
    }
    else {
        tensorflow::Tensor preds = output[0];
        
        auto preds_map = preds.tensor<float, 2>();
        
        //for (int j = 0; j < SEQ_LEN; j++){ // SEQ_LEN for model trained with shift_pred as True.
        for (int j = 0; j < 1; j++){ // 1 for model trained with shift_pred as False, and pred_len = 1
            NSLog(@"%f = %f", preds_map(0,j), scale * preds_map(0,j) + lower);
            txt = [NSString stringWithFormat:@"%@\nPrediction with Keras RNN model:\n%f", txt, scale * preds_map(0,j) + lower];
            dispatch_async(dispatch_get_main_queue(), ^{
                [_tv setText:txt];
                [_tv sizeToFit];
            });
            
        }
    }
    
}


- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.

    _tv = [[UITextView alloc] initWithFrame:CGRectMake(50, 50, 300, 300)];
    self.automaticallyAdjustsScrollViewInsets = NO;
    [self.view addSubview:_tv];
    _tv.textColor = [UIColor blackColor];
    _tv.font = [UIFont systemFontOfSize:16];
    
    _btn = [UIButton buttonWithType:UIButtonTypeSystem];
    [_btn setTranslatesAutoresizingMaskIntoConstraints:NO];
    _btn.titleLabel.font = [UIFont systemFontOfSize:32];
    [_btn setTitle:@"Predict" forState:UIControlStateNormal];
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


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
