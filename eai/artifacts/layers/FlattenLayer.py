CONTRACT_ARTIFACT = {   '_format': 'hh-sol-artifact-1',
    'abi': [   {   'inputs': [   {   'internalType': 'bytes',
                                     'name': 'config',
                                     'type': 'bytes'}],
                   'stateMutability': 'nonpayable',
                   'type': 'constructor'},
               {'inputs': [], 'name': 'IncorrectTensorType', 'type': 'error'},
               {   'inputs': [],
                   'name': 'TensorTypeNotSupported',
                   'type': 'error'},
               {   'inputs': [   {   'internalType': 'Float32x32[]',
                                     'name': 'x',
                                     'type': 'int64[]'},
                                 {   'internalType': 'uint256',
                                     'name': 'idx',
                                     'type': 'uint256'}],
                   'name': 'appendWeights',
                   'outputs': [   {   'internalType': 'uint256',
                                      'name': '',
                                      'type': 'uint256'},
                                  {   'internalType': 'bool',
                                      'name': '',
                                      'type': 'bool'}],
                   'stateMutability': 'nonpayable',
                   'type': 'function'},
               {   'inputs': [   {   'components': [   {   'internalType': 'bytes',
                                                           'name': 'data',
                                                           'type': 'bytes'},
                                                       {   'internalType': 'uint256[]',
                                                           'name': 'dim',
                                                           'type': 'uint256[]'}],
                                     'internalType': 'struct '
                                                     'Tensors.TensorData[]',
                                     'name': 'input',
                                     'type': 'tuple[]'}],
                   'name': 'forward',
                   'outputs': [   {   'components': [   {   'internalType': 'bytes',
                                                            'name': 'data',
                                                            'type': 'bytes'},
                                                        {   'internalType': 'uint256[]',
                                                            'name': 'dim',
                                                            'type': 'uint256[]'}],
                                      'internalType': 'struct '
                                                      'Tensors.TensorData',
                                      'name': '',
                                      'type': 'tuple'}],
                   'stateMutability': 'view',
                   'type': 'function'},
               {   'inputs': [],
                   'name': 'getParamsCount',
                   'outputs': [   {   'internalType': 'uint256',
                                      'name': '',
                                      'type': 'uint256'}],
                   'stateMutability': 'view',
                   'type': 'function'}],
    'bytecode': '0x60806040523480156200001157600080fd5b506040516200146838038062001468833981016040819052620000349162000051565b5062000126565b634e487b7160e01b600052604160045260246000fd5b600060208083850312156200006557600080fd5b82516001600160401b03808211156200007d57600080fd5b818501915085601f8301126200009257600080fd5b815181811115620000a757620000a76200003b565b604051601f8201601f19908116603f01168101908382118183101715620000d257620000d26200003b565b816040528281528886848701011115620000eb57600080fd5b600093505b828410156200010f5784840186015181850187015292850192620000f0565b600086848301015280965050505050505092915050565b61133280620001366000396000f3fe608060405234801561001057600080fd5b50600436106100415760003560e01c80635c0cf0f414610046578063612934491461005a578063768e789714610086575b600080fd5b604051600081526020015b60405180910390f35b610071610068366004610bf0565b92600192509050565b60408051928352901515602083015201610051565b610099610094366004610c3c565b6100a6565b6040516100519190610c7e565b6040805180820190915260608082526020820152828260008181106100cd576100cd610d1d565b90506020028101906100df9190610d33565b6100ed906020810190610d71565b9050600003610128576040517f035a803f00000000000000000000000000000000000000000000000000000000815260040160405180910390fd5b8282600081811061013b5761013b610d1d565b905060200281019061014d9190610d33565b61015b906020810190610d71565b9050600103610199578282600081811061017757610177610d1d565b90506020028101906101899190610d33565b61019290610eba565b90506103d3565b828260008181106101ac576101ac610d1d565b90506020028101906101be9190610d33565b6101cc906020810190610d71565b9050600203610279576000838360008181106101ea576101ea610d1d565b90506020028101906101fc9190610d33565b6102069080610f78565b81019061021391906110a6565b90506000610220826103d9565b9050600061022d8261042e565b90506040518060400160405280826000015160405160200161024f91906110e3565b604051602081830303815290604052815260200161026c83610475565b81525093505050506103d3565b8282600081811061028c5761028c610d1d565b905060200281019061029e9190610d33565b6102ac906020810190610d71565b905060030361030d576000838360008181106102ca576102ca610d1d565b90506020028101906102dc9190610d33565b6102e69080610f78565b8101906102f3919061119e565b90506000610300826104c0565b9050600061022d8261055b565b8282600081811061032057610320610d1d565b90506020028101906103329190610d33565b610340906020810190610d71565b90506004036103a15760008383600081811061035e5761035e610d1d565b90506020028101906103709190610d33565b61037a9080610f78565b81019061038791906111d3565b9050600061039482610580565b9050600061022d8261067b565b6040517f6801957400000000000000000000000000000000000000000000000000000000815260040160405180910390fd5b92915050565b6103fd60405180606001604052806060815260200160008152602001600081525090565b815160208201528151829060009061041757610417610d1d565b602090810291909101015151604082015290815290565b6040805180820190915260608152600060208201526103d361045383600001516106a0565b6040805180820190915260608152600060208201528151602082015290815290565b604080516001808252818301909252606091602080830190803683370190505090508160200151816000815181106104af576104af610d1d565b602002602001018181525050919050565b6104eb6040518060800160405280606081526020016000815260200160008152602001600081525090565b815160208201528151829060009061050557610505610d1d565b6020026020010151518160400181815250508160008151811061052a5761052a610d1d565b602002602001015160008151811061054457610544610d1d565b602090810291909101015151606082015290815290565b6040805180820190915260608152600060208201526103d361045383600001516107d5565b6105b26040518060a0016040528060608152602001600081526020016000815260200160008152602001600081525090565b81516020820152815182906000906105cc576105cc610d1d565b602002602001015151816040018181525050816000815181106105f1576105f1610d1d565b602002602001015160008151811061060b5761060b610d1d565b6020026020010151518160600181815250508160008151811061063057610630610d1d565b602002602001015160008151811061064a5761064a610d1d565b602002602001015160008151811061066457610664610d1d565b602090810291909101015151608082015290815290565b6040805180820190915260608152600060208201526103d36104538360000151610971565b60606000826000815181106106b7576106b7610d1d565b60200260200101515183516106cc919061129a565b67ffffffffffffffff8111156106e4576106e4610dbb565b60405190808252806020026020018201604052801561070d578160200160208202803683370190505b5090506000805b84518110156107cc5760005b85828151811061073257610732610d1d565b6020026020010151518110156107b95785828151811061075457610754610d1d565b6020026020010151818151811061076d5761076d610d1d565b602002602001015184848151811061078757610787610d1d565b60079290920b602092830291909101909101526107a56001846112b1565b9250806107b1816112c4565b915050610720565b50806107c4816112c4565b915050610714565b50909392505050565b805160609060008381836107eb576107eb610d1d565b602002602001015151905060008460008151811061080b5761080b610d1d565b602002602001015160008151811061082557610825610d1d565b6020026020010151519050600081838561083f919061129a565b610849919061129a565b67ffffffffffffffff81111561086157610861610dbb565b60405190808252806020026020018201604052801561088a578160200160208202803683370190505b5090506000805b858110156109655760005b858110156109525760005b8581101561093f578983815181106108c1576108c1610d1d565b602002602001015182815181106108da576108da610d1d565b602002602001015181815181106108f3576108f3610d1d565b602002602001015185858151811061090d5761090d610d1d565b60079290920b6020928302919091019091015261092b6001856112b1565b935080610937816112c4565b9150506108a7565b508061094a816112c4565b91505061089c565b508061095d816112c4565b915050610891565b50909695505050505050565b8051606090600083818361098757610987610d1d565b60200260200101515190506000846000815181106109a7576109a7610d1d565b60200260200101516000815181106109c1576109c1610d1d565b60200260200101515190506000856000815181106109e1576109e1610d1d565b60200260200101516000815181106109fb576109fb610d1d565b6020026020010151600081518110610a1557610a15610d1d565b6020026020010151519050600081838587610a30919061129a565b610a3a919061129a565b610a44919061129a565b67ffffffffffffffff811115610a5c57610a5c610dbb565b604051908082528060200260200182016040528015610a85578160200160208202803683370190505b5090506000805b86811015610b975760005b86811015610b845760005b86811015610b715760005b86811015610b5e578b8481518110610ac757610ac7610d1d565b60200260200101518381518110610ae057610ae0610d1d565b60200260200101518281518110610af957610af9610d1d565b60200260200101518181518110610b1257610b12610d1d565b6020026020010151868681518110610b2c57610b2c610d1d565b60079290920b60209283029190910190910152610b4a6001866112b1565b945080610b56816112c4565b915050610aad565b5080610b69816112c4565b915050610aa2565b5080610b7c816112c4565b915050610a97565b5080610b8f816112c4565b915050610a8c565b5090979650505050505050565b60008083601f840112610bb657600080fd5b50813567ffffffffffffffff811115610bce57600080fd5b6020830191508360208260051b8501011115610be957600080fd5b9250929050565b600080600060408486031215610c0557600080fd5b833567ffffffffffffffff811115610c1c57600080fd5b610c2886828701610ba4565b909790965060209590950135949350505050565b60008060208385031215610c4f57600080fd5b823567ffffffffffffffff811115610c6657600080fd5b610c7285828601610ba4565b90969095509350505050565b600060208083528351604082850152805180606086015260005b81811015610cb457828101840151868201608001528301610c98565b506000858201608090810182905287850151601f909301601f1916870187810360600160408901528351918101829052838601945091929160a001905b80841015610d115784518252938501936001939093019290850190610cf1565b50979650505050505050565b634e487b7160e01b600052603260045260246000fd5b600082357fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffc1833603018112610d6757600080fd5b9190910192915050565b6000808335601e19843603018112610d8857600080fd5b83018035915067ffffffffffffffff821115610da357600080fd5b6020019150600581901b3603821315610be957600080fd5b634e487b7160e01b600052604160045260246000fd5b6040805190810167ffffffffffffffff81118282101715610df457610df4610dbb565b60405290565b604051601f8201601f1916810167ffffffffffffffff81118282101715610e2357610e23610dbb565b604052919050565b600067ffffffffffffffff821115610e4557610e45610dbb565b5060051b60200190565b600082601f830112610e6057600080fd5b81356020610e75610e7083610e2b565b610dfa565b82815260059290921b84018101918181019086841115610e9457600080fd5b8286015b84811015610eaf5780358352918301918301610e98565b509695505050505050565b600060408236031215610ecc57600080fd5b610ed4610dd1565b823567ffffffffffffffff80821115610eec57600080fd5b9084019036601f830112610eff57600080fd5b8135602082821115610f1357610f13610dbb565b610f2581601f19601f85011601610dfa565b8281523682848701011115610f3957600080fd5b82828601838301376000818401830152855286810135935082841115610f5e57600080fd5b610f6a36858901610e4f565b908501525091949350505050565b6000808335601e19843603018112610f8f57600080fd5b83018035915067ffffffffffffffff821115610faa57600080fd5b602001915036819003821315610be957600080fd5b600082601f830112610fd057600080fd5b81356020610fe0610e7083610e2b565b828152600592831b8501820192828201919087851115610fff57600080fd5b8387015b85811015610b9757803567ffffffffffffffff8111156110235760008081fd5b8801603f81018a136110355760008081fd5b858101356040611047610e7083610e2b565b82815291851b8301810191888101908d8411156110645760008081fd5b938201935b8385101561109557843592508260070b83146110855760008081fd5b8282529389019390890190611069565b885250505093850193508401611003565b6000602082840312156110b857600080fd5b813567ffffffffffffffff8111156110cf57600080fd5b6110db84828501610fbf565b949350505050565b6020808252825182820181905260009190848201906040850190845b8181101561096557835160070b835292840192918401916001016110ff565b600082601f83011261112f57600080fd5b8135602061113f610e7083610e2b565b82815260059290921b8401810191818101908684111561115e57600080fd5b8286015b84811015610eaf57803567ffffffffffffffff8111156111825760008081fd5b6111908986838b0101610fbf565b845250918301918301611162565b6000602082840312156111b057600080fd5b813567ffffffffffffffff8111156111c757600080fd5b6110db8482850161111e565b600060208083850312156111e657600080fd5b823567ffffffffffffffff808211156111fe57600080fd5b818501915085601f83011261121257600080fd5b8135611220610e7082610e2b565b81815260059190911b8301840190848101908883111561123f57600080fd5b8585015b838110156112775780358581111561125b5760008081fd5b6112698b89838a010161111e565b845250918601918601611243565b5098975050505050505050565b634e487b7160e01b600052601160045260246000fd5b80820281158282048414176103d3576103d3611284565b808201808211156103d3576103d3611284565b60007fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff82036112f5576112f5611284565b506001019056fea2646970667358221220ad3aae3c83a76934ee781224e45971626769a40e3f303d25a8b3d2393f2e9f5564736f6c63430008130033',
    'contractName': 'FlattenLayer',
    'deployedBytecode': '0x608060405234801561001057600080fd5b50600436106100415760003560e01c80635c0cf0f414610046578063612934491461005a578063768e789714610086575b600080fd5b604051600081526020015b60405180910390f35b610071610068366004610bf0565b92600192509050565b60408051928352901515602083015201610051565b610099610094366004610c3c565b6100a6565b6040516100519190610c7e565b6040805180820190915260608082526020820152828260008181106100cd576100cd610d1d565b90506020028101906100df9190610d33565b6100ed906020810190610d71565b9050600003610128576040517f035a803f00000000000000000000000000000000000000000000000000000000815260040160405180910390fd5b8282600081811061013b5761013b610d1d565b905060200281019061014d9190610d33565b61015b906020810190610d71565b9050600103610199578282600081811061017757610177610d1d565b90506020028101906101899190610d33565b61019290610eba565b90506103d3565b828260008181106101ac576101ac610d1d565b90506020028101906101be9190610d33565b6101cc906020810190610d71565b9050600203610279576000838360008181106101ea576101ea610d1d565b90506020028101906101fc9190610d33565b6102069080610f78565b81019061021391906110a6565b90506000610220826103d9565b9050600061022d8261042e565b90506040518060400160405280826000015160405160200161024f91906110e3565b604051602081830303815290604052815260200161026c83610475565b81525093505050506103d3565b8282600081811061028c5761028c610d1d565b905060200281019061029e9190610d33565b6102ac906020810190610d71565b905060030361030d576000838360008181106102ca576102ca610d1d565b90506020028101906102dc9190610d33565b6102e69080610f78565b8101906102f3919061119e565b90506000610300826104c0565b9050600061022d8261055b565b8282600081811061032057610320610d1d565b90506020028101906103329190610d33565b610340906020810190610d71565b90506004036103a15760008383600081811061035e5761035e610d1d565b90506020028101906103709190610d33565b61037a9080610f78565b81019061038791906111d3565b9050600061039482610580565b9050600061022d8261067b565b6040517f6801957400000000000000000000000000000000000000000000000000000000815260040160405180910390fd5b92915050565b6103fd60405180606001604052806060815260200160008152602001600081525090565b815160208201528151829060009061041757610417610d1d565b602090810291909101015151604082015290815290565b6040805180820190915260608152600060208201526103d361045383600001516106a0565b6040805180820190915260608152600060208201528151602082015290815290565b604080516001808252818301909252606091602080830190803683370190505090508160200151816000815181106104af576104af610d1d565b602002602001018181525050919050565b6104eb6040518060800160405280606081526020016000815260200160008152602001600081525090565b815160208201528151829060009061050557610505610d1d565b6020026020010151518160400181815250508160008151811061052a5761052a610d1d565b602002602001015160008151811061054457610544610d1d565b602090810291909101015151606082015290815290565b6040805180820190915260608152600060208201526103d361045383600001516107d5565b6105b26040518060a0016040528060608152602001600081526020016000815260200160008152602001600081525090565b81516020820152815182906000906105cc576105cc610d1d565b602002602001015151816040018181525050816000815181106105f1576105f1610d1d565b602002602001015160008151811061060b5761060b610d1d565b6020026020010151518160600181815250508160008151811061063057610630610d1d565b602002602001015160008151811061064a5761064a610d1d565b602002602001015160008151811061066457610664610d1d565b602090810291909101015151608082015290815290565b6040805180820190915260608152600060208201526103d36104538360000151610971565b60606000826000815181106106b7576106b7610d1d565b60200260200101515183516106cc919061129a565b67ffffffffffffffff8111156106e4576106e4610dbb565b60405190808252806020026020018201604052801561070d578160200160208202803683370190505b5090506000805b84518110156107cc5760005b85828151811061073257610732610d1d565b6020026020010151518110156107b95785828151811061075457610754610d1d565b6020026020010151818151811061076d5761076d610d1d565b602002602001015184848151811061078757610787610d1d565b60079290920b602092830291909101909101526107a56001846112b1565b9250806107b1816112c4565b915050610720565b50806107c4816112c4565b915050610714565b50909392505050565b805160609060008381836107eb576107eb610d1d565b602002602001015151905060008460008151811061080b5761080b610d1d565b602002602001015160008151811061082557610825610d1d565b6020026020010151519050600081838561083f919061129a565b610849919061129a565b67ffffffffffffffff81111561086157610861610dbb565b60405190808252806020026020018201604052801561088a578160200160208202803683370190505b5090506000805b858110156109655760005b858110156109525760005b8581101561093f578983815181106108c1576108c1610d1d565b602002602001015182815181106108da576108da610d1d565b602002602001015181815181106108f3576108f3610d1d565b602002602001015185858151811061090d5761090d610d1d565b60079290920b6020928302919091019091015261092b6001856112b1565b935080610937816112c4565b9150506108a7565b508061094a816112c4565b91505061089c565b508061095d816112c4565b915050610891565b50909695505050505050565b8051606090600083818361098757610987610d1d565b60200260200101515190506000846000815181106109a7576109a7610d1d565b60200260200101516000815181106109c1576109c1610d1d565b60200260200101515190506000856000815181106109e1576109e1610d1d565b60200260200101516000815181106109fb576109fb610d1d565b6020026020010151600081518110610a1557610a15610d1d565b6020026020010151519050600081838587610a30919061129a565b610a3a919061129a565b610a44919061129a565b67ffffffffffffffff811115610a5c57610a5c610dbb565b604051908082528060200260200182016040528015610a85578160200160208202803683370190505b5090506000805b86811015610b975760005b86811015610b845760005b86811015610b715760005b86811015610b5e578b8481518110610ac757610ac7610d1d565b60200260200101518381518110610ae057610ae0610d1d565b60200260200101518281518110610af957610af9610d1d565b60200260200101518181518110610b1257610b12610d1d565b6020026020010151868681518110610b2c57610b2c610d1d565b60079290920b60209283029190910190910152610b4a6001866112b1565b945080610b56816112c4565b915050610aad565b5080610b69816112c4565b915050610aa2565b5080610b7c816112c4565b915050610a97565b5080610b8f816112c4565b915050610a8c565b5090979650505050505050565b60008083601f840112610bb657600080fd5b50813567ffffffffffffffff811115610bce57600080fd5b6020830191508360208260051b8501011115610be957600080fd5b9250929050565b600080600060408486031215610c0557600080fd5b833567ffffffffffffffff811115610c1c57600080fd5b610c2886828701610ba4565b909790965060209590950135949350505050565b60008060208385031215610c4f57600080fd5b823567ffffffffffffffff811115610c6657600080fd5b610c7285828601610ba4565b90969095509350505050565b600060208083528351604082850152805180606086015260005b81811015610cb457828101840151868201608001528301610c98565b506000858201608090810182905287850151601f909301601f1916870187810360600160408901528351918101829052838601945091929160a001905b80841015610d115784518252938501936001939093019290850190610cf1565b50979650505050505050565b634e487b7160e01b600052603260045260246000fd5b600082357fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffc1833603018112610d6757600080fd5b9190910192915050565b6000808335601e19843603018112610d8857600080fd5b83018035915067ffffffffffffffff821115610da357600080fd5b6020019150600581901b3603821315610be957600080fd5b634e487b7160e01b600052604160045260246000fd5b6040805190810167ffffffffffffffff81118282101715610df457610df4610dbb565b60405290565b604051601f8201601f1916810167ffffffffffffffff81118282101715610e2357610e23610dbb565b604052919050565b600067ffffffffffffffff821115610e4557610e45610dbb565b5060051b60200190565b600082601f830112610e6057600080fd5b81356020610e75610e7083610e2b565b610dfa565b82815260059290921b84018101918181019086841115610e9457600080fd5b8286015b84811015610eaf5780358352918301918301610e98565b509695505050505050565b600060408236031215610ecc57600080fd5b610ed4610dd1565b823567ffffffffffffffff80821115610eec57600080fd5b9084019036601f830112610eff57600080fd5b8135602082821115610f1357610f13610dbb565b610f2581601f19601f85011601610dfa565b8281523682848701011115610f3957600080fd5b82828601838301376000818401830152855286810135935082841115610f5e57600080fd5b610f6a36858901610e4f565b908501525091949350505050565b6000808335601e19843603018112610f8f57600080fd5b83018035915067ffffffffffffffff821115610faa57600080fd5b602001915036819003821315610be957600080fd5b600082601f830112610fd057600080fd5b81356020610fe0610e7083610e2b565b828152600592831b8501820192828201919087851115610fff57600080fd5b8387015b85811015610b9757803567ffffffffffffffff8111156110235760008081fd5b8801603f81018a136110355760008081fd5b858101356040611047610e7083610e2b565b82815291851b8301810191888101908d8411156110645760008081fd5b938201935b8385101561109557843592508260070b83146110855760008081fd5b8282529389019390890190611069565b885250505093850193508401611003565b6000602082840312156110b857600080fd5b813567ffffffffffffffff8111156110cf57600080fd5b6110db84828501610fbf565b949350505050565b6020808252825182820181905260009190848201906040850190845b8181101561096557835160070b835292840192918401916001016110ff565b600082601f83011261112f57600080fd5b8135602061113f610e7083610e2b565b82815260059290921b8401810191818101908684111561115e57600080fd5b8286015b84811015610eaf57803567ffffffffffffffff8111156111825760008081fd5b6111908986838b0101610fbf565b845250918301918301611162565b6000602082840312156111b057600080fd5b813567ffffffffffffffff8111156111c757600080fd5b6110db8482850161111e565b600060208083850312156111e657600080fd5b823567ffffffffffffffff808211156111fe57600080fd5b818501915085601f83011261121257600080fd5b8135611220610e7082610e2b565b81815260059190911b8301840190848101908883111561123f57600080fd5b8585015b838110156112775780358581111561125b5760008081fd5b6112698b89838a010161111e565b845250918601918601611243565b5098975050505050505050565b634e487b7160e01b600052601160045260246000fd5b80820281158282048414176103d3576103d3611284565b808201808211156103d3576103d3611284565b60007fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff82036112f5576112f5611284565b506001019056fea2646970667358221220ad3aae3c83a76934ee781224e45971626769a40e3f303d25a8b3d2393f2e9f5564736f6c63430008130033',
    'deployedLinkReferences': {},
    'linkReferences': {},
    'sourceName': 'contracts/lib/layers-new/FlattenLayer.sol'}