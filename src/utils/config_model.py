import model.transformer_wireneigh as transformer_wireneigh
import model.transformer_clora as transformer_clora
import model.transformer_wirefix as transformer_wirefix
import model.transformer_original as transformer_original
import model.baselines.transformer_adapter as transformer_adapter
import model.baselines.transformer_pt as transformer_pt
import model.baselines.transformer_pv2 as transformer_pv2
import model.baselines.transformer_l2p as transformer_l2p
import model.baselines.transformer_coda as transformer_coda

def load_method(method, vocabulary, class_all, kwargs, if_decoding, n_tasks):
    print("Loading the " + method + " model...")
    #############################
    #Our global alignment method
    #############################
    if method == 'wirefix':
        config = transformer_wirefix.TransformerConfig(vocabulary, class_all, kwargs)
        model = transformer_wirefix.Transformer(config, class_head = True, mlm_head = if_decoding) 
    elif method == 'wireneigh':
        config = transformer_wireneigh.TransformerConfig(vocabulary, class_all, kwargs)
        model = transformer_wireneigh.Transformer(config, class_head = True, mlm_head = if_decoding) 
    elif method == 'clora':
        config = transformer_clora.TransformerConfig(vocabulary, class_all, kwargs)
        model = transformer_clora.Transformer(config, class_head = True, mlm_head = if_decoding) 
    #############################
    #Baseline methods
    #############################
    elif method == 'origin':
        #Load the original fine-tuning model
        config = transformer_original.TransformerConfig(vocabulary, class_all, kwargs)
        model = transformer_original.Transformer(config, class_head = True, mlm_head = if_decoding)
    elif method == 'adapter':
        config = transformer_adapter.TransformerConfig(vocabulary, class_all, kwargs)
        model = transformer_adapter.Transformer(config, class_head = True, mlm_head = if_decoding)
    elif method == 'pt':
        config = transformer_pt.TransformerConfig(vocabulary, class_all, kwargs)
        model = transformer_pt.Transformer(config, class_head = True, mlm_head = if_decoding)
    elif method == 'pv2':
        config = transformer_pv2.TransformerConfig(vocabulary, class_all, kwargs)
        model = transformer_pv2.Transformer(config, class_head = True, mlm_head = if_decoding)
    elif method == 'l2p':
        config = transformer_l2p.TransformerConfig(vocabulary, class_all, kwargs)
        model = transformer_l2p.Transformer(config, class_head = True, mlm_head = if_decoding)
    elif method == 'coda':
        config = transformer_coda.TransformerConfig(vocabulary, class_all, kwargs)
        config.n_tasks = n_tasks
        model = transformer_coda.Transformer(config, class_head = True, mlm_head = if_decoding)
    return config, model