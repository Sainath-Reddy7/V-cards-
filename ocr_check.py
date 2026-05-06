try:
    import pytesseract
    from PIL import Image
    print('pytesseract-ok')
except Exception as e:
    print('pytesseract-missing', repr(e))
