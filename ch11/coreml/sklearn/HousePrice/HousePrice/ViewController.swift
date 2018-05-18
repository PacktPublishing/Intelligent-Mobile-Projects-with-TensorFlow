//
//  ViewController.swift
//  HousePrice
//
//  Created by Jeff Tang on 3/17/18.
//  Copyright © 2018 Jeff Tang. All rights reserved.
//

import UIKit

class ViewController: UIViewController {
    private let lr = HouseLR()
    private let svm = HouseSVM()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        let lr_input = HouseLRInput(Bedrooms: 3, Bathrooms: 2, Size: 1560)
        let svm_input = HouseSVMInput(Bedrooms: 3, Bathrooms: 2, Size: 1560)
        
        guard let lr_output = try? lr.prediction(input: lr_input) else {
            return
        }
        
        print(lr_output.Price)
        
        guard let svm_output = try? svm.prediction(input: svm_input) else {
            return
        }
        
        print(svm_output.Price)

    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


}

