#include <iostream>
#include <fstream>
#include <memory>
#include <cstring>

int main()
{
    std::cout << "\n\n------------------------------ Start Write File ------------------------------\n\n";

    static constexpr size_t size = 1024*1024*2;
    std::cout << "File size: " << size << "\n\n";

    {
        std::cout << "File0: 0\n";
        std::ofstream outfile("/mnt/tmpfs/file_check0.bin");
        std::unique_ptr<char[]> buffer = std::make_unique<char[]>(size);
        memset(buffer.get(), 0x00, size);

        outfile.write(buffer.get(), size);
        outfile.close();
    }

    {
        std::cout << "File1: F\n";
        std::ofstream outfile("/mnt/tmpfs/file_checkF.bin");
        std::unique_ptr<char[]> buffer = std::make_unique<char[]>(size);
        memset(buffer.get(), 0xFF, size);

        outfile.write(buffer.get(), size);
        outfile.close();
    }

    {
        std::cout << "File2: 5\n";
        std::ofstream outfile("/mnt/tmpfs/file_check5.bin");
        std::unique_ptr<char[]> buffer = std::make_unique<char[]>(size);
        memset(buffer.get(), 0x55, size);

        outfile.write(buffer.get(), size);
        outfile.close();
    }

    {
        std::cout << "File3: A\n";
        std::ofstream outfile("/mnt/tmpfs/file_checkA.bin");
        std::unique_ptr<char[]> buffer = std::make_unique<char[]>(size);
        memset(buffer.get(), 0xAA, size);

        outfile.write(buffer.get(), size);
        outfile.close();
    }
    std::cout << "\n\n------------------------------ End Write File ------------------------------\n\n";

    return 0;
}
