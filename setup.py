from distutils.core import setup, Extension
import sysconfig

def main():
    CFLAGS = ['-g', '-Wall', '-std=c99', '-fopenmp', '-mavx', '-mfma', '-pthread', '-O3']
    LDFLAGS = ['-fopenmp']
    # Use the setup function we imported and set up the modules.
    # You may find this reference helpful: https://docs.python.org/3.6/extending/building.html
    # TODO: YOUR CODE HERE
    # raise NotImplementedError("You need to complete task 2 to install your module!")
    module1 = Extension("numc",
                        sources = ['src/numc.c', 'src/matrix.c'],
                        include_dirs = ['/usr/include/python3'],
                        extra_compile_args = CFLAGS,
                        extra_link_args = LDFLAGS)
    
    setup(name = 'numc',
            version = '1.0',
            description = 'numc matrix package',
            author = 'Logan Chiu',
            author_email = 'logan.c.chiu@berkeley.edu',
            ext_modules = [module1])

if __name__ == "__main__":
    main()
