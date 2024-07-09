CONTRACT_ARTIFACT = {'_format': 'hh-sol-artifact-1',
                     'abi': [{'inputs': [{'internalType': 'bytes',
                                          'name': 'config',
                                          'type': 'bytes'}],
                              'stateMutability': 'nonpayable',
                              'type': 'constructor'},
                             {'inputs': [{'internalType': 'Float32x32[]',
                                          'name': 'x',
                                          'type': 'int64[]'},
                                         {'internalType': 'uint256',
                                          'name': 'idx',
                                          'type': 'uint256'}],
                              'name': 'appendWeights',
                              'outputs': [{'internalType': 'uint256',
                                           'name': '',
                                           'type': 'uint256'},
                                          {'internalType': 'bool',
                                           'name': '',
                                           'type': 'bool'}],
                              'stateMutability': 'nonpayable',
                              'type': 'function'},
                             {'inputs': [{'components': [{'internalType': 'bytes',
                                                          'name': 'data',
                                                          'type': 'bytes'},
                                                         {'internalType': 'uint256[]',
                                                          'name': 'dim',
                                                          'type': 'uint256[]'}],
                                          'internalType': 'struct '
                                          'Tensors.TensorData[]',
                                          'name': 'input',
                                          'type': 'tuple[]'}],
                              'name': 'forward',
                              'outputs': [{'components': [{'internalType': 'bytes',
                                                           'name': 'data',
                                                           'type': 'bytes'},
                                                          {'internalType': 'uint256[]',
                                                           'name': 'dim',
                                                           'type': 'uint256[]'}],
                                           'internalType': 'struct '
                                           'Tensors.TensorData',
                                           'name': '',
                                           'type': 'tuple'}],
                              'stateMutability': 'view',
                              'type': 'function'},
                             {'inputs': [],
                              'name': 'getInputDim',
                              'outputs': [{'internalType': 'uint256[]',
                                           'name': '',
                                           'type': 'uint256[]'}],
                              'stateMutability': 'view',
                              'type': 'function'},
                             {'inputs': [],
                              'name': 'getParamsCount',
                              'outputs': [{'internalType': 'uint256',
                                           'name': '',
                                           'type': 'uint256'}],
                              'stateMutability': 'view',
                              'type': 'function'}],
                     'bytecode': '0x60806040523480156200001157600080fd5b50604051620008953803806200089583398101604081905262000034916200015d565b600080828060200190518101906200004d919062000213565b8051919350915062000067906001906020840190620000ad565b508160ff166003811115620000805762000080620002e1565b6000805460ff191660018360038111156200009f576200009f620002e1565b0217905550505050620002f7565b828054828255906000526020600020908101928215620000eb579160200282015b82811115620000eb578251825591602001919060010190620000ce565b50620000f9929150620000fd565b5090565b5b80821115620000f95760008155600101620000fe565b634e487b7160e01b600052604160045260246000fd5b604051601f8201601f191681016001600160401b038111828210171562000155576200015562000114565b604052919050565b600060208083850312156200017157600080fd5b82516001600160401b03808211156200018957600080fd5b818501915085601f8301126200019e57600080fd5b815181811115620001b357620001b362000114565b620001c7601f8201601f191685016200012a565b91508082528684828501011115620001de57600080fd5b60005b81811015620001fe578381018501518382018601528401620001e1565b50600090820190930192909252509392505050565b600080604083850312156200022757600080fd5b825160ff811681146200023957600080fd5b602084810151919350906001600160401b03808211156200025957600080fd5b818601915086601f8301126200026e57600080fd5b81518181111562000283576200028362000114565b8060051b9150620002968483016200012a565b8181529183018401918481019089841115620002b157600080fd5b938501935b83851015620002d157845182529385019390850190620002b6565b8096505050505050509250929050565b634e487b7160e01b600052602160045260246000fd5b61058e80620003076000396000f3fe608060405234801561001057600080fd5b506004361061004c5760003560e01c80635c0cf0f4146100515780636129344914610065578063763ada6014610091578063768e7897146100a6575b600080fd5b604051600081526020015b60405180910390f35b61007c6100733660046101b3565b92600192509050565b6040805192835290151560208301520161005c565b6100996100c6565b60405161005c91906101ff565b6100b96100b4366004610243565b61011e565b60405161005c9190610285565b6060600180548060200260200160405190810160405280929190818152602001828054801561011457602002820191906000526020600020905b815481526020019060010190808311610100575b5050505050905090565b60408051808201909152606080825260208201528282600081811061014557610145610324565b90506020028101906101579190610353565b6101609061049a565b9392505050565b60008083601f84011261017957600080fd5b50813567ffffffffffffffff81111561019157600080fd5b6020830191508360208260051b85010111156101ac57600080fd5b9250929050565b6000806000604084860312156101c857600080fd5b833567ffffffffffffffff8111156101df57600080fd5b6101eb86828701610167565b909790965060209590950135949350505050565b6020808252825182820181905260009190848201906040850190845b818110156102375783518352928401929184019160010161021b565b50909695505050505050565b6000806020838503121561025657600080fd5b823567ffffffffffffffff81111561026d57600080fd5b61027985828601610167565b90969095509350505050565b600060208083528351604082850152805180606086015260005b818110156102bb5782810184015186820160800152830161029f565b506000858201608090810182905287850151601f909301601f1916870187810360600160408901528351918101829052838601945091929160a001905b8084101561031857845182529385019360019390930192908501906102f8565b50979650505050505050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052603260045260246000fd5b600082357fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffc183360301811261038757600080fd5b9190910192915050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052604160045260246000fd5b6040805190810167ffffffffffffffff811182821017156103e3576103e3610391565b60405290565b604051601f8201601f1916810167ffffffffffffffff8111828210171561041257610412610391565b604052919050565b600082601f83011261042b57600080fd5b8135602067ffffffffffffffff82111561044757610447610391565b8160051b6104568282016103e9565b928352848101820192828101908785111561047057600080fd5b83870192505b8483101561048f57823582529183019190830190610476565b979650505050505050565b6000604082360312156104ac57600080fd5b6104b46103c0565b823567ffffffffffffffff808211156104cc57600080fd5b9084019036601f8301126104df57600080fd5b81356020828211156104f3576104f3610391565b61050581601f19601f850116016103e9565b828152368284870101111561051957600080fd5b8282860183830137600081840183015285528681013593508284111561053e57600080fd5b61054a3685890161041a565b90850152509194935050505056fea2646970667358221220eedb39c23a534e25f23112c89c568a351753301868faf494b412a0a51583771164736f6c63430008130033',
                     'contractName': 'InputLayer',
                     'deployedBytecode': '0x608060405234801561001057600080fd5b506004361061004c5760003560e01c80635c0cf0f4146100515780636129344914610065578063763ada6014610091578063768e7897146100a6575b600080fd5b604051600081526020015b60405180910390f35b61007c6100733660046101b3565b92600192509050565b6040805192835290151560208301520161005c565b6100996100c6565b60405161005c91906101ff565b6100b96100b4366004610243565b61011e565b60405161005c9190610285565b6060600180548060200260200160405190810160405280929190818152602001828054801561011457602002820191906000526020600020905b815481526020019060010190808311610100575b5050505050905090565b60408051808201909152606080825260208201528282600081811061014557610145610324565b90506020028101906101579190610353565b6101609061049a565b9392505050565b60008083601f84011261017957600080fd5b50813567ffffffffffffffff81111561019157600080fd5b6020830191508360208260051b85010111156101ac57600080fd5b9250929050565b6000806000604084860312156101c857600080fd5b833567ffffffffffffffff8111156101df57600080fd5b6101eb86828701610167565b909790965060209590950135949350505050565b6020808252825182820181905260009190848201906040850190845b818110156102375783518352928401929184019160010161021b565b50909695505050505050565b6000806020838503121561025657600080fd5b823567ffffffffffffffff81111561026d57600080fd5b61027985828601610167565b90969095509350505050565b600060208083528351604082850152805180606086015260005b818110156102bb5782810184015186820160800152830161029f565b506000858201608090810182905287850151601f909301601f1916870187810360600160408901528351918101829052838601945091929160a001905b8084101561031857845182529385019360019390930192908501906102f8565b50979650505050505050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052603260045260246000fd5b600082357fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffc183360301811261038757600080fd5b9190910192915050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052604160045260246000fd5b6040805190810167ffffffffffffffff811182821017156103e3576103e3610391565b60405290565b604051601f8201601f1916810167ffffffffffffffff8111828210171561041257610412610391565b604052919050565b600082601f83011261042b57600080fd5b8135602067ffffffffffffffff82111561044757610447610391565b8160051b6104568282016103e9565b928352848101820192828101908785111561047057600080fd5b83870192505b8483101561048f57823582529183019190830190610476565b979650505050505050565b6000604082360312156104ac57600080fd5b6104b46103c0565b823567ffffffffffffffff808211156104cc57600080fd5b9084019036601f8301126104df57600080fd5b81356020828211156104f3576104f3610391565b61050581601f19601f850116016103e9565b828152368284870101111561051957600080fd5b8282860183830137600081840183015285528681013593508284111561053e57600080fd5b61054a3685890161041a565b90850152509194935050505056fea2646970667358221220eedb39c23a534e25f23112c89c568a351753301868faf494b412a0a51583771164736f6c63430008130033',
                     'deployedLinkReferences': {},
                     'linkReferences': {},
                     'sourceName': 'contracts/lib/layers-new/InputLayer.sol'}