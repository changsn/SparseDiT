import torch
import pdb

def load_model(target, source, target_modules, source_modules, verbose=False):
    missing_keys = []
    for key in target:
        flag = 1
        for module in target_modules:
            if key.startswith(module) and key[len(module)] == '.':
                flag = 0
                source_key = source_modules[target_modules.index(module)] + key[len(module):]
                if source_key in source:
                    target[key] = source[source_key]
                    del source[source_key]
                else:
                    if '.attn.q.' in key:
                        source_key = source_key.replace('.attn.q.', '.attn.qkv.')
                        if source_key in source:
                            target[key] = source[source_key].chunk(3, dim=0)[0]
                        else:
                            missing_keys.append(key)
                    elif '.attn.v.' in key:
                        source_key = source_key.replace('.attn.v.', '.attn.qkv.')
                        if source_key in source:
                            target[key] = source[source_key].chunk(3, dim=0)[2]
                        else:
                            missing_keys.append(key)
                    elif 'attn.kv.'in key:
                        source_key = source_key.replace('.attn.kv.', '.attn.qkv.')
                        if source_key in source:
                            target[key] = source[source_key][source[source_key].size(0)//3:]
                        else:
                            missing_keys.append(key)
                    else:
                        missing_keys.append(key)
                break
            else:
                pass
        if flag:
            if key in source:
                target[key] = source[key]
                del source[key]
            else:
                missing_keys.append(key)
    unexpected_keys = list(source.keys())
    if verbose:
        return target, missing_keys, unexpected_keys
    else:
        return target
    

        