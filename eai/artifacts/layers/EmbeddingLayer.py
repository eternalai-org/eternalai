CONTRACT_ARTIFACT = {'_format': 'hh-sol-artifact-1',
 'abi': [{'inputs': [{'internalType': 'bytes',
                      'name': 'config',
                      'type': 'bytes'}],
          'stateMutability': 'nonpayable',
          'type': 'constructor'},
         {'inputs': [], 'name': 'IncorrectTensorType', 'type': 'error'},
         {'inputs': [], 'name': 'TensorTypeNotSupported', 'type': 'error'},
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
                      {'internalType': 'bool', 'name': '', 'type': 'bool'}],
          'stateMutability': 'nonpayable',
          'type': 'function'},
         {'inputs': [{'components': [{'internalType': 'bytes',
                                      'name': 'data',
                                      'type': 'bytes'},
                                     {'internalType': 'uint256[]',
                                      'name': 'dim',
                                      'type': 'uint256[]'}],
                      'internalType': 'struct Tensors.TensorData[]',
                      'name': 'input',
                      'type': 'tuple[]'}],
          'name': 'forward',
          'outputs': [{'components': [{'internalType': 'bytes',
                                       'name': 'data',
                                       'type': 'bytes'},
                                      {'internalType': 'uint256[]',
                                       'name': 'dim',
                                       'type': 'uint256[]'}],
                       'internalType': 'struct Tensors.TensorData',
                       'name': '',
                       'type': 'tuple'}],
          'stateMutability': 'view',
          'type': 'function'},
         {'inputs': [],
          'name': 'getParamsCount',
          'outputs': [{'internalType': 'uint256',
                       'name': '',
                       'type': 'uint256'}],
          'stateMutability': 'view',
          'type': 'function'},
         {'inputs': [{'internalType': 'uint256',
                      'name': 'i',
                      'type': 'uint256'},
                     {'internalType': 'uint256',
                      'name': 'j',
                      'type': 'uint256'}],
          'name': 'getWeight',
          'outputs': [{'internalType': 'Float32x32',
                       'name': '',
                       'type': 'int64'}],
          'stateMutability': 'view',
          'type': 'function'},
         {'inputs': [],
          'name': 'inputDim',
          'outputs': [{'internalType': 'uint256',
                       'name': '',
                       'type': 'uint256'}],
          'stateMutability': 'view',
          'type': 'function'},
         {'inputs': [],
          'name': 'outputDim',
          'outputs': [{'internalType': 'uint256',
                       'name': '',
                       'type': 'uint256'}],
          'stateMutability': 'view',
          'type': 'function'},
         {'inputs': [],
          'name': 'w',
          'outputs': [{'internalType': 'uint256',
                       'name': 'n',
                       'type': 'uint256'},
                      {'internalType': 'uint256',
                       'name': 'm',
                       'type': 'uint256'}],
          'stateMutability': 'view',
          'type': 'function'}],
 'bytecode': '0x60806040523480156200001157600080fd5b5060405162001095380380620010958339810160408190526200003491620002c7565b600080828060200190518101906200004d91906200039c565b600082905560018190559092509050620000688282620000a1565b8051805160029162000080918391602001906200012d565b506020820151816001015560408201518160020155905050505050620003c1565b620000c660405180606001604052806060815260200160008152602001600081525090565b6020810183905260408101829052826001600160401b03811115620000ef57620000ef620002b1565b6040519080825280602002602001820160405280156200012457816020015b60608152602001906001900390816200010e5790505b50815292915050565b8280548282559060005260206000209081019282156200017f579160200282015b828111156200017f57825180516200016e91849160209091019062000191565b50916020019190600101906200014e565b506200018d9291506200024f565b5090565b82805482825590600052602060002090600301600490048101928215620002415791602002820160005b838211156200020a57835183826101000a8154816001600160401b03021916908360070b6001600160401b031602179055509260200192600801602081600701049283019260010302620001bb565b80156200023f5782816101000a8154906001600160401b0302191690556008016020816007010492830192600103026200020a565b505b506200018d92915062000270565b808211156200018d57600062000266828262000287565b506001016200024f565b5b808211156200018d576000815560010162000271565b508054600082556003016004900490600052602060002090810190620002ae919062000270565b50565b634e487b7160e01b600052604160045260246000fd5b60006020808385031215620002db57600080fd5b82516001600160401b0380821115620002f357600080fd5b818501915085601f8301126200030857600080fd5b8151818111156200031d576200031d620002b1565b604051601f8201601f19908116603f01168101908382118183101715620003485762000348620002b1565b8160405282815288868487010111156200036157600080fd5b600093505b8284101562000385578484018601518185018701529285019262000366565b600086848301015280965050505050505092915050565b60008060408385031215620003b057600080fd5b505080516020909101519092909150565b610cc480620003d16000396000f3fe608060405234801561001057600080fd5b506004361061007d5760003560e01c80635c0cf0f41161005b5780635c0cf0f4146100e757806361293449146100ef57806372356fd614610117578063768e78971461012057600080fd5b80630f39ee7214610082578063205c9cc7146100ad5780634dc0d879146100d0575b600080fd5b6100956100903660046107c8565b610140565b60405160079190910b81526020015b60405180910390f35b6003546004546100bb919082565b604080519283526020830191909152016100a4565b6100d960005481565b6040519081526020016100a4565b6100d961019f565b6101026100fd366004610836565b610284565b604080519283529015156020830152016100a4565b6100d960015481565b61013361012e366004610882565b61039f565b6040516100a491906108c4565b60006002600001838154811061015857610158610963565b90600052602060002001828154811061017357610173610963565b90600052602060002090600491828204019190066008029054906101000a900460070b90505b92915050565b6040805160028054608060208202840181019094526060830181815260009461027f949392849291849190889085015b8282101561025e5760008481526020908190208301805460408051828502810185019091528181529283018282801561024a57602002820191906000526020600020906000905b825461010083900a900460070b81526020600f83018190049384019360010360089093019290920291018084116102165790505b5050505050815260200190600101906101cf565b505050508152602001600182015481526020016002820154815250506105e3565b905090565b6005546006546000918291818303610387576004546003546000906102aa90839061098f565b90505b87871080156102bb57508083105b1561036d5760026102cc83856109a6565b815481106102dc576102dc610963565b906000526020600020018989898181106102f8576102f8610963565b905060200201602081019061030d91906109df565b8154600181018355600092835260209092206004830401805460039093166008026101000a67ffffffffffffffff8181021990941692909316929092021790558261035781610a01565b935050868061036590610a01565b9750506102ad565b8083036103845761037d84610a01565b9350600092505b50505b60058290556006558392506001149050935093915050565b6040805180820190915260608082526020820152828260008181106103c6576103c6610963565b90506020028101906103d89190610a39565b6103e6906020810190610a77565b9050600003610421576040517f035a803f00000000000000000000000000000000000000000000000000000000815260040160405180910390fd5b8282600081811061043457610434610963565b90506020028101906104469190610a39565b610454906020810190610a77565b90506001036105b15760008383600081811061047257610472610963565b90506020028101906104849190610a39565b61048e9080610ac1565b81019061049b9190610b1e565b905060006002600001805480602002602001604051908101604052809291908181526020016000905b828210156105535760008481526020908190208301805460408051828502810185019091528181529283018282801561053f57602002820191906000526020600020906000905b825461010083900a900460070b81526020600f830181900493840193600103600890930192909202910180841161050b5790505b5050505050815260200190600101906104c4565b505050509050600061056582846105f9565b9050604051806040016040528082600001516040516020016105879190610be3565b60405160208183030381529060405281526020016105a483610706565b8152509350505050610199565b6040517f6801957400000000000000000000000000000000000000000000000000000000815260040160405180910390fd5b600081604001518260200151610199919061098f565b61061d60405180606001604052806060815260200160008152602001600081525090565b6000825167ffffffffffffffff81111561063957610639610b08565b60405190808252806020026020018201604052801561066c57816020015b60608152602001906001900390816106575790505b50905060005b83518110156106f457846106a285838151811061069157610691610963565b602002602001015160070b60201d90565b67ffffffffffffffff16815181106106bc576106bc610963565b60200260200101518282815181106106d6576106d6610963565b602002602001018190525080806106ec90610a01565b915050610672565b506106fe81610773565b949350505050565b604080516002808252606080830184529260208301908036833701905050905081602001518160008151811061073e5761073e610963565b60200260200101818152505081604001518160018151811061076257610762610963565b602002602001018181525050919050565b61079760405180606001604052806060815260200160008152602001600081525090565b81516020820152815182906000906107b1576107b1610963565b602090810291909101015151604082015290815290565b600080604083850312156107db57600080fd5b50508035926020909101359150565b60008083601f8401126107fc57600080fd5b50813567ffffffffffffffff81111561081457600080fd5b6020830191508360208260051b850101111561082f57600080fd5b9250929050565b60008060006040848603121561084b57600080fd5b833567ffffffffffffffff81111561086257600080fd5b61086e868287016107ea565b909790965060209590950135949350505050565b6000806020838503121561089557600080fd5b823567ffffffffffffffff8111156108ac57600080fd5b6108b8858286016107ea565b90969095509350505050565b600060208083528351604082850152805180606086015260005b818110156108fa578281018401518682016080015283016108de565b506000858201608090810182905287850151601f909301601f1916870187810360600160408901528351918101829052838601945091929160a001905b808410156109575784518252938501936001939093019290850190610937565b50979650505050505050565b634e487b7160e01b600052603260045260246000fd5b634e487b7160e01b600052601160045260246000fd5b808202811582820484141761019957610199610979565b6000826109c357634e487b7160e01b600052601260045260246000fd5b500490565b8035600781900b81146109da57600080fd5b919050565b6000602082840312156109f157600080fd5b6109fa826109c8565b9392505050565b60007fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff8203610a3257610a32610979565b5060010190565b600082357fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffc1833603018112610a6d57600080fd5b9190910192915050565b6000808335601e19843603018112610a8e57600080fd5b83018035915067ffffffffffffffff821115610aa957600080fd5b6020019150600581901b360382131561082f57600080fd5b6000808335601e19843603018112610ad857600080fd5b83018035915067ffffffffffffffff821115610af357600080fd5b60200191503681900382131561082f57600080fd5b634e487b7160e01b600052604160045260246000fd5b60006020808385031215610b3157600080fd5b823567ffffffffffffffff80821115610b4957600080fd5b818501915085601f830112610b5d57600080fd5b813581811115610b6f57610b6f610b08565b8060051b604051601f19603f83011681018181108582111715610b9457610b94610b08565b604052918252848201925083810185019188831115610bb257600080fd5b938501935b82851015610bd757610bc8856109c8565b84529385019392850192610bb7565b98975050505050505050565b6000602080830181845280855180835260408601915060408160051b87010192508387016000805b83811015610c80578886037fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffc0018552825180518088529088019088880190845b81811015610c6a57835160070b8352928a0192918a0191600101610c4b565b5090975050509386019391860191600101610c0b565b50939897505050505050505056fea2646970667358221220d841886c305aff63dee9a4094c79aa2cc2b9bd618bb5f57cd431d27acb90fb2464736f6c63430008130033',
 'contractName': 'EmbeddingLayer',
 'deployedBytecode': '0x608060405234801561001057600080fd5b506004361061007d5760003560e01c80635c0cf0f41161005b5780635c0cf0f4146100e757806361293449146100ef57806372356fd614610117578063768e78971461012057600080fd5b80630f39ee7214610082578063205c9cc7146100ad5780634dc0d879146100d0575b600080fd5b6100956100903660046107c8565b610140565b60405160079190910b81526020015b60405180910390f35b6003546004546100bb919082565b604080519283526020830191909152016100a4565b6100d960005481565b6040519081526020016100a4565b6100d961019f565b6101026100fd366004610836565b610284565b604080519283529015156020830152016100a4565b6100d960015481565b61013361012e366004610882565b61039f565b6040516100a491906108c4565b60006002600001838154811061015857610158610963565b90600052602060002001828154811061017357610173610963565b90600052602060002090600491828204019190066008029054906101000a900460070b90505b92915050565b6040805160028054608060208202840181019094526060830181815260009461027f949392849291849190889085015b8282101561025e5760008481526020908190208301805460408051828502810185019091528181529283018282801561024a57602002820191906000526020600020906000905b825461010083900a900460070b81526020600f83018190049384019360010360089093019290920291018084116102165790505b5050505050815260200190600101906101cf565b505050508152602001600182015481526020016002820154815250506105e3565b905090565b6005546006546000918291818303610387576004546003546000906102aa90839061098f565b90505b87871080156102bb57508083105b1561036d5760026102cc83856109a6565b815481106102dc576102dc610963565b906000526020600020018989898181106102f8576102f8610963565b905060200201602081019061030d91906109df565b8154600181018355600092835260209092206004830401805460039093166008026101000a67ffffffffffffffff8181021990941692909316929092021790558261035781610a01565b935050868061036590610a01565b9750506102ad565b8083036103845761037d84610a01565b9350600092505b50505b60058290556006558392506001149050935093915050565b6040805180820190915260608082526020820152828260008181106103c6576103c6610963565b90506020028101906103d89190610a39565b6103e6906020810190610a77565b9050600003610421576040517f035a803f00000000000000000000000000000000000000000000000000000000815260040160405180910390fd5b8282600081811061043457610434610963565b90506020028101906104469190610a39565b610454906020810190610a77565b90506001036105b15760008383600081811061047257610472610963565b90506020028101906104849190610a39565b61048e9080610ac1565b81019061049b9190610b1e565b905060006002600001805480602002602001604051908101604052809291908181526020016000905b828210156105535760008481526020908190208301805460408051828502810185019091528181529283018282801561053f57602002820191906000526020600020906000905b825461010083900a900460070b81526020600f830181900493840193600103600890930192909202910180841161050b5790505b5050505050815260200190600101906104c4565b505050509050600061056582846105f9565b9050604051806040016040528082600001516040516020016105879190610be3565b60405160208183030381529060405281526020016105a483610706565b8152509350505050610199565b6040517f6801957400000000000000000000000000000000000000000000000000000000815260040160405180910390fd5b600081604001518260200151610199919061098f565b61061d60405180606001604052806060815260200160008152602001600081525090565b6000825167ffffffffffffffff81111561063957610639610b08565b60405190808252806020026020018201604052801561066c57816020015b60608152602001906001900390816106575790505b50905060005b83518110156106f457846106a285838151811061069157610691610963565b602002602001015160070b60201d90565b67ffffffffffffffff16815181106106bc576106bc610963565b60200260200101518282815181106106d6576106d6610963565b602002602001018190525080806106ec90610a01565b915050610672565b506106fe81610773565b949350505050565b604080516002808252606080830184529260208301908036833701905050905081602001518160008151811061073e5761073e610963565b60200260200101818152505081604001518160018151811061076257610762610963565b602002602001018181525050919050565b61079760405180606001604052806060815260200160008152602001600081525090565b81516020820152815182906000906107b1576107b1610963565b602090810291909101015151604082015290815290565b600080604083850312156107db57600080fd5b50508035926020909101359150565b60008083601f8401126107fc57600080fd5b50813567ffffffffffffffff81111561081457600080fd5b6020830191508360208260051b850101111561082f57600080fd5b9250929050565b60008060006040848603121561084b57600080fd5b833567ffffffffffffffff81111561086257600080fd5b61086e868287016107ea565b909790965060209590950135949350505050565b6000806020838503121561089557600080fd5b823567ffffffffffffffff8111156108ac57600080fd5b6108b8858286016107ea565b90969095509350505050565b600060208083528351604082850152805180606086015260005b818110156108fa578281018401518682016080015283016108de565b506000858201608090810182905287850151601f909301601f1916870187810360600160408901528351918101829052838601945091929160a001905b808410156109575784518252938501936001939093019290850190610937565b50979650505050505050565b634e487b7160e01b600052603260045260246000fd5b634e487b7160e01b600052601160045260246000fd5b808202811582820484141761019957610199610979565b6000826109c357634e487b7160e01b600052601260045260246000fd5b500490565b8035600781900b81146109da57600080fd5b919050565b6000602082840312156109f157600080fd5b6109fa826109c8565b9392505050565b60007fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff8203610a3257610a32610979565b5060010190565b600082357fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffc1833603018112610a6d57600080fd5b9190910192915050565b6000808335601e19843603018112610a8e57600080fd5b83018035915067ffffffffffffffff821115610aa957600080fd5b6020019150600581901b360382131561082f57600080fd5b6000808335601e19843603018112610ad857600080fd5b83018035915067ffffffffffffffff821115610af357600080fd5b60200191503681900382131561082f57600080fd5b634e487b7160e01b600052604160045260246000fd5b60006020808385031215610b3157600080fd5b823567ffffffffffffffff80821115610b4957600080fd5b818501915085601f830112610b5d57600080fd5b813581811115610b6f57610b6f610b08565b8060051b604051601f19603f83011681018181108582111715610b9457610b94610b08565b604052918252848201925083810185019188831115610bb257600080fd5b938501935b82851015610bd757610bc8856109c8565b84529385019392850192610bb7565b98975050505050505050565b6000602080830181845280855180835260408601915060408160051b87010192508387016000805b83811015610c80578886037fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffc0018552825180518088529088019088880190845b81811015610c6a57835160070b8352928a0192918a0191600101610c4b565b5090975050509386019391860191600101610c0b565b50939897505050505050505056fea2646970667358221220d841886c305aff63dee9a4094c79aa2cc2b9bd618bb5f57cd431d27acb90fb2464736f6c63430008130033',
 'deployedLinkReferences': {},
 'linkReferences': {},
 'sourceName': 'contracts/lib/layers-new/EmbeddingLayer.sol'}