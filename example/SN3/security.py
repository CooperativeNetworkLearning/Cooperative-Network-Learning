# -*- coding: utf-8 -*-
import json
from base64 import b64encode, b64decode
import rsa
import pickle
# 创建公钥和私钥文件
def createKeyFile(keySize, filePath):
    runStatus = False
    try:
        private, public = rsa.newkeys(keySize)
        with open(f'{filePath}/public.pem', 'wb') as f:
            f.write(public.save_pkcs1())
        with open(f'{filePath}/private.pem', 'wb') as f:
            f.write(private.save_pkcs1())

        print(f'private: {private.save_pkcs1("PEM")}')
        print(f'public: {public.save_pkcs1("PEM")}')
        runStatus = True
    except Exception as e:
        print(e.args[0])

    return runStatus

# 获取公钥
def loadPublicKey(filePath):
    with open(filePath, 'rb') as f:
        pubKey = rsa.PublicKey.load_pkcs1(f.read())
    return pubKey

# 获取私钥
def loadPrivateKey(filePath):
    with open(filePath, 'rb') as f:
        privkey = rsa.PrivateKey.load_pkcs1(f.read())
    return privkey

# 公钥加密
def encrypt(text, pubKey, charset='utf-8'):
    if not isinstance(text, bytes):
        data = text.encode(charset)
    else:
        data = text
    length = len(data)
    default_length = 53
    res = []
    for i in range(0, length, default_length):
        res.append(rsa.encrypt(data[i:i + default_length], pubKey))
    byte_data = b''.join(res)
    return b64encode(byte_data)
    # return b64encode(rsa.encrypt(text, pubKey))

# 私钥解秘
def decrypt(ciphertext, privkey):
    data = b64decode(ciphertext)
    length = len(data)
    default_length = 64
    res = []
    for i in range(0, length, default_length):
        res.append(rsa.decrypt(data[i:i + default_length], privkey))
    return b''.join(res)    
    # return str(b''.join(res), encoding="utf-8")
    # return rsa.decrypt(b64decode(ciphertext), privkey)

# 私钥签名
def sign(text, privkey, hashAlg="SHA-256"):
    return b64encode(rsa.sign(text, privkey, hashAlg))


# 公钥验签
def verify(text, signature, pub_key):
    signatureVerify = False
    try:
        rsa.verify(text, b64decode(signature), pub_key)
        signatureVerify = True
    except rsa.VerificationError as e:
        print(e.args[0])
    return signatureVerify


def encrypt_by_name(text, pubKey_name, charset='utf-8'):
    text = pickle.dumps(text)
    pubKey = loadPublicKey('./public/'+pubKey_name+'.pem')
    if not isinstance(text, bytes):
        data = text.encode(charset)
    else:
        data = text
    length = len(data)
    default_length = 53
    res = []
    for i in range(0, length, default_length):
        res.append(rsa.encrypt(data[i:i + default_length], pubKey))
    byte_data = b''.join(res)
    return b64encode(byte_data)
def encrypt_self(text, charset='utf-8'):
    text = pickle.dumps(text)
    pubKey = loadPublicKey('./public.pem')
    if not isinstance(text, bytes):
        data = text.encode(charset)
    else:
        data = text
    length = len(data)
    default_length = 53
    res = []
    for i in range(0, length, default_length):
        res.append(rsa.encrypt(data[i:i + default_length], pubKey))
    byte_data = b''.join(res)
    return b64encode(byte_data)
def decrypt(ciphertext):
    privkey = loadPrivateKey('./private.pem')
    data = b64decode(ciphertext)
    length = len(data)
    default_length = 64
    res = []
    for i in range(0, length, default_length):
        res.append(rsa.decrypt(data[i:i + default_length], privkey))
    res = b''.join(res)
    return pickle.loads(res)