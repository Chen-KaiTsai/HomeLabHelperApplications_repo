#include <iostream>
#include <fstream>
#include <memory>
#include <cstring>
#include <omp.h>

#include <thread>
#include <chrono>
#include <atomic>

std::atomic<bool> stop_flag;

void count_bits(char a, char b, size_t &StoUS, size_t &UStoS)
{
    uint count = 0;
    bool a_bit, b_bit;
    for (size_t i = 0; i < sizeof(char) * 8; ++i) {
        a_bit = (a >> i) & 1;
        b_bit = (b >> i) & 1;

        if (b_bit != a_bit) {
            StoUS += b_bit;
            UStoS += a_bit;
        }
    }
}

void host_stopper()
{
    getchar();
    stop_flag = true;
}

int main()
{
    omp_set_num_threads(2);

    size_t index = 0;
    stop_flag = false;

    std::thread stopper(host_stopper);

    while (true)
    {
        if(stop_flag)
            break;

        std::cout << "\n\n------------------------------ Start Checking File #" << index << " ------------------------------\n\n";

        #pragma omp parallel sections
        {
	    std::cout << "Number of thread invoked: " << omp_get_num_threads() << std::endl;
            #pragma omp section
            {
                std::ifstream inputfile("./file_check0.bin");

                size_t StoUS = 0;
                size_t UStoS = 0;

                if(inputfile.is_open()) {
                    char byte = inputfile.get();
                    while (inputfile.good()) {
                        count_bits(byte ,0x00, StoUS, UStoS);
                        byte = inputfile.get();
                    }
                }
                else {
                    std::cout << "ERROR: Unalbe to open file\n";
                }
                inputfile.close();

                #pragma omp critical
                {
                    std::cout << "File0: 0\n";
                    std::cout << "0 -> 1: " << UStoS << "\n1 -> 0: " << StoUS << "\n\n";
                }
            }

            #pragma omp section
            {
                std::ifstream inputfile("./file_checkF.bin");

                size_t StoUS = 0;
                size_t UStoS = 0;

                if(inputfile.is_open()) {
                    char byte = inputfile.get();
                    while (inputfile.good()) {
                        count_bits(byte ,0xFF, StoUS, UStoS);
                        byte = inputfile.get();
                    }
                }
                else {
                    std::cout << "ERROR: Unalbe to open file\n";
                }
                inputfile.close();
                #pragma omp critical
                {
                    std::cout << "File1: F\n";
                    std::cout << "0 -> 1: " << UStoS << "\n1 -> 0: " << StoUS << "\n\n";
                }
            }

            #pragma omp section
            {
                std::ifstream inputfile("./file_check5.bin");

                size_t StoUS = 0;
                size_t UStoS = 0;

                if(inputfile.is_open()) {
                    char byte = inputfile.get();
                    while (inputfile.good()) {
                        count_bits(byte ,0x55, StoUS, UStoS);
                        byte = inputfile.get();
                    }
                }
                else {
                    std::cout << "ERROR: Unalbe to open file\n";
                }
                inputfile.close();

                #pragma omp critical
                {
                    std::cout << "File2: 5\n";
                    std::cout << "0 -> 1: " << UStoS << "\n1 -> 0: " << StoUS << "\n\n";
                }
            }

            #pragma omp section
            {
                std::ifstream inputfile("./file_checkA.bin");

                size_t StoUS = 0;
                size_t UStoS = 0;

                if(inputfile.is_open()) {
                    char byte = inputfile.get();
                    while (inputfile.good()) {
                        count_bits(byte ,0xAA, StoUS, UStoS);
                        byte = inputfile.get();
                    }
                }
                else {
                    std::cout << "ERROR: Unalbe to open file\n";
                }
                inputfile.close();

                #pragma omp critical
                {
                    std::cout << "File3: A\n";
                    std::cout << "0 -> 1: " << UStoS << "\n1 -> 0: " << StoUS << "\n\n";
                }
            }
        }
        std::cout << "------------------------------ End Checking File #" << index++ << " ------------------------------\n\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
    }

    stopper.join();
    return 0;
}
