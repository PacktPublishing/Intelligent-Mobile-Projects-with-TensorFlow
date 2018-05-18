//
//  RunInference.h
//  AudioRecognition_Swift
//
//  Created by Jeff Tang on 1/25/18.
//  Copyright © 2018 Jeff Tang. All rights reserved.
//
#import <Foundation/Foundation.h>

@interface RunInference_Wrapper : NSObject
- (NSString *)run_inference_wrapper:(NSString*)recorderFilePath;
@end


