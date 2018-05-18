import coremltools

coreml_model = coremltools.converters.keras.convert('stock.h5', input_names=['bidirectional_1_input'], output_names=['activation_1/Identity'])
coreml_model.save('Stock.mlmodel')
