#include <iostream>
#include <fstream>
#include <string>
#include <list>
#include <deque>
#include <map>
#include <thread>
#include <future>
#include <boost/compute.hpp>
#include "synthSignal/Signal.hpp"
#include "synthSignal/SineWaveform.hpp"
#include "minusDarwin/Solver.hpp"
#include <Utility.hpp>


namespace bc = boost::compute;

const unsigned int sps = 8000;
const float a = 127.0f*sqrt(2);
const float omega = 2.0f*M_PI*60.0f;
const float phi = M_PI;
const float omegaMin = 2.0f*M_PI*60.0f*0.94f;
const float omegaMax = 2.0f*M_PI*60.0f*1.06f;
const float phiMax = 2.0f*M_PI;
const size_t maxGens = 10;
const size_t replications = 10;

struct cmpDevice{
    bool operator()(const bc::device& lhs, const bc::device &rhs) {
        return lhs.id() < rhs.id();
    }
};
const char sinewaveFitSource[] =
        BOOST_COMPUTE_STRINGIZE_SOURCE(
                __kernel void calculateScoresOfPopulation(
                        __global const float *signalData,
                __global const float2 *population,
                __global float *scores,
                const unsigned int populationSize,
                const unsigned int signalDataSize,
                const float sumOfSquares,
                const float omegaMin,
                const float omegaMax,
                const float phiMax,
                const unsigned int sps,
                const float a)
        {
                const uint aId = get_global_id(0);
                const float2 agent = population[aId];
                float error = 0.0f;
                for (size_t p = 0; p < signalDataSize; p++) {
                float t = (float)p/(float)sps;
                float realOmega = omegaMin+agent.x*(omegaMax-omegaMin);
                float realPhi = agent.y*phiMax;
                float estimated =
                a*sin(realOmega*t+realPhi);
                error += (estimated-signalData[p])*(estimated-signalData[p]);
        }
                scores[aId] = error/sumOfSquares;
        });
const std::string outputFilenameParallel = "SinewaveTimeParallel.csv";
const std::string outputFilenameSerial = "SinewaveTimeSerial.csv";
int main() {
    std::cout << "Initializing test..." << std::endl;
    std::ofstream ofParallel(outputFilenameParallel.c_str(),std::ofstream::out);
    std::ofstream ofSerial(outputFilenameSerial.c_str(),std::ofstream::out);
    std::vector<size_t> popSizeVector({25,50,100,200});
    std::vector<float> signalTimeVector({1.0f,2.0f,4.0f,8.0f});
    ofParallel << "popSize,signalTime,mean,sd" << std::endl;
    ofSerial << "popSize,signalTime,mean,sd" << std::endl;

    auto device = bc::system::default_device();
    auto ctx = new bc::context(device);
    auto queue = new bc::command_queue(*ctx,device);
    auto program = bc::program::create_with_source(sinewaveFitSource, *ctx);
    // compile the program
    try {
        program.build();
    } catch(bc::opencl_error &e) {
        std::cout << program.build_log() << std::endl;
    }
    auto kernel = new bc::kernel(program,"calculateScoresOfPopulation");
    //kernel->set_arg(0,*dSignal);
    //kernel->set_arg(1,*dX);
    //kernel->set_arg(2,*dScores);
    //kernel->set_arg(3,(unsigned int)dX->size());
    //kernel->set_arg(4,(unsigned int)dSignal->size());
    //kernel->set_arg(5,sumOfSquares);
    kernel->set_arg(6,omegaMin);
    kernel->set_arg(7,omegaMax);
    kernel->set_arg(8,phiMax);
    kernel->set_arg(9,(unsigned int)sps);
    kernel->set_arg(10,a);
    for(auto popSize : popSizeVector)
        for(auto signalTime : signalTimeVector) {
            SynthSignal::Signal signalModel;
            SynthSignal::Interpolation interpolation({},SynthSignal::InterpolationType::LINEAR);
            interpolation.addPoint(0.0f,1.0f);
            interpolation.addPoint(signalTime,1.0f);
            SynthSignal::Interpolation frequencyVariation({},SynthSignal::InterpolationType::LINEAR);
            frequencyVariation.addPoint(0.0f,1.0f);
            frequencyVariation.addPoint(signalTime,1.0f);
            auto wf = new SynthSignal::SineWaveform(
                    a,
                    omega,
                    phi,
                    frequencyVariation
            );
            signalModel.addEvent(wf,interpolation);
            auto signal = signalModel.gen(signalTime,sps);
            auto dSignal = new bc::vector<float>(
                    signal->begin(),signal->end(),*queue);
            const float sumOfSquares =
                    std::accumulate(signal->begin(),signal->end(),0.0f,
                                    [](float accum,float val) -> float {
                                        accum = accum+val*val;
                                      });
            auto dX = new bc::vector<bc::float2_>(popSize,*ctx);
            auto dScores = new bc::vector<float>(popSize,*ctx);

            kernel->set_arg(0,*dSignal);
            kernel->set_arg(1,*dX);
            kernel->set_arg(2,*dScores);
            kernel->set_arg(3,(unsigned int)dX->size());
            kernel->set_arg(4,(unsigned int)dSignal->size());
            kernel->set_arg(5,sumOfSquares);

            auto evaluatePopulationLambdaParallel =
                    [queue, kernel, dSignal, dScores, dX, sumOfSquares](
                            MinusDarwin::Scores &scores,
                            const MinusDarwin::Population &population) {
                        //Population to Device
                        std::vector<bc::float2_> hPopulation(population.size());
                        std::transform(population.begin(),population.end(),
                                       hPopulation.begin(),[](const MinusDarwin::Agent &a){
                                    bc::float2_ b;
                                    b[0] = a.at(0);
                                    b[1] = a.at(1);
                                    return b;
                                });
                        bc::copy(hPopulation.begin(),hPopulation.end(),dX->begin(),*queue);
                        //Once population has been copied to the device
                        //a parallel calculation of scores must be done
                        queue->enqueue_1d_range_kernel(*kernel,0,hPopulation.size(),0);
                        bc::copy(dScores->begin(),dScores->end(),scores.begin(),*queue);
                    };


            auto evaluatePopulationLambdaSerial =
                    [signal, sumOfSquares/*, sps, a,
                            omegaMin, omegaMax, phiMax*/](
                            MinusDarwin::Scores &scores,
                            const MinusDarwin::Population &population) {
                        for (size_t aId = 0; aId < population.size(); ++aId) {
                            auto &agent = population.at(aId);
                            float error = 0.0f;
                            for (size_t p = 0; p < signal->size(); p++) {
                                float t = (float)p/(float)sps;
                                float realOmega = omegaMin+agent.at(0)*(omegaMax-omegaMin);
                                float realPhi = agent.at(1)*phiMax;
                                float estimated =
                                        a*sin(realOmega*t+realPhi);
                                error += pow(estimated-signal->at(p),2.0f);
                            }
                            scores.at(aId) = error/sumOfSquares;
                        }
                    };

            MinusDarwin::SolverParameterSet config = {
                    2,
                    popSize,
                    maxGens,
                    MinusDarwin::GoalFunction::MaxGenerations,
                    MinusDarwin::CrossoverMode::Best,
                    1,
                    0.005f,
                    1,
                    0.25f,
                    true
            };
            std::vector<long long> timeVectorParallel(replications,0);
            std::vector<long long> timeVectorSerial(replications,0);
            for(size_t replication = 0; replication < replications; replication++) {
                MinusDarwin::Solver solverParallel(config,evaluatePopulationLambdaParallel);
                auto tStart = boost::chrono::high_resolution_clock::now();
                solverParallel.run(false);
                auto tEnd = boost::chrono::high_resolution_clock::now();
                auto tDuration = boost::chrono::duration_cast<boost::chrono::milliseconds>(tEnd-tStart);
                long long tCount = tDuration.count();
                timeVectorParallel.at(replication) = tCount;

                MinusDarwin::Solver solverSerial(config,evaluatePopulationLambdaSerial);
                tStart = boost::chrono::high_resolution_clock::now();
                solverSerial.run(false);
                tEnd = boost::chrono::high_resolution_clock::now();
                tDuration = boost::chrono::duration_cast<boost::chrono::milliseconds>(tEnd-tStart);
                tCount = tDuration.count();
                timeVectorSerial.at(replication) = tCount;
            }
            float accum1Parallel = std::accumulate(timeVectorParallel.begin(),timeVectorParallel.end(),0.0f);
            float meanParallel = accum1Parallel/(float)replications;
            float accum2Parallel =
                    std::accumulate(timeVectorParallel.begin(),timeVectorParallel.end(),0.0f,[&meanParallel](float accum,float val) {
                        return accum+pow(val-meanParallel,2);
                    });
            float sdParallel = sqrt(accum2Parallel/(float)replications);
            ofParallel << popSize << "," << signalTime << "," << meanParallel << "," << sdParallel << std::endl;
            float accum1Serial = std::accumulate(timeVectorSerial.begin(),timeVectorSerial.end(),0.0f);
            float meanSerial = accum1Serial/(float)replications;
            float accum2Serial =
                    std::accumulate(timeVectorSerial.begin(),timeVectorSerial.end(),0.0f,[&meanSerial](float accum,float val) {
                        return accum+pow(val-meanSerial,2);
                    });
            float sdSerial = sqrt(accum2Serial/(float)replications);
            ofSerial << popSize << "," << signalTime << "," << meanSerial << "," << sdSerial << std::endl;
            //TODO Up
            delete(dSignal);
            delete(dX);
            delete(dScores);
            delete(wf);
        }
    ofParallel.close();
    ofSerial.close();
    return 0;
}