/* SerialParallelComparison.cpp
 * This application will produce an output
 * containing time results from running 80
 * generations using the general optimum
 * configuration for Sinewave fitting.
 * Results output is in CSV format that
 * will be managed by R in order to produce
 * a figure output at the end.
 */
#include <iostream>
#include <fstream>
#include <string>
#include <list>
#include <deque>
#include <map>
#include <thread>
#include <future>
#include <boost/compute.hpp>
#include <boost/chrono.hpp>
#include "synthSignal/Signal.hpp"
#include "synthSignal/SineWaveform.hpp"
#include "minusDarwin/Solver.hpp"
#include <Utility.hpp>


namespace bc = boost::compute;
struct ExperimentConfiguration {
    float signalTime;
    size_t rep;
};
typedef std::pair<std::vector<float> *, float> SignalAndSS;
typedef std::pair<ExperimentConfiguration, long long> ResultPair;

const std::string outputFilename = "SerialParallelComparison.csv";
const std::vector<float> signalTimeVector = {1.0f,2.0f,4.0f,8.0f/*,16.0f,32.0f*/};
const unsigned int sps = 8000;
const float a = 127.0f*sqrt(2);
const float omega = 2.0f*M_PI*60.0f;
const float phi = M_PI;
const float omegaMin = 2.0f*M_PI*60.0f*0.94f;
const float omegaMax = 2.0f*M_PI*60.0f*1.06f;
const float phiMax = 2.0f*M_PI;
const size_t replications = 10;
MinusDarwin::SolverParameterSet config = {
        2,
        /*200*/200,
        /*80*/20,
        MinusDarwin::GoalFunction::MaxGenerations,
        MinusDarwin::CrossoverMode::Best,
        1,
        0.005f,
        1.0f,
        0.25f,
        true
};


struct cmpDevice{
    bool operator()(const bc::device& lhs, const bc::device &rhs) {
        return lhs.id() < rhs.id();
    }
};
const char sinewaveFitSourceParallel[] =
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

std::vector<ResultPair> runGPUExperiments(const std::vector<SignalAndSS> &signalsAndSSVector);
std::vector<ResultPair> runCPUExperiments(const std::vector<SignalAndSS> &signalsAndSSVector);
std::vector<SignalAndSS> createSignalsVector();
int main(int argc, char **argv) {
    std::cout << "[SerialParallelComparison] Initializing test..." << std::endl;
    std::ofstream of(outputFilename.c_str(), std::ofstream::out);
    // Mode: Parallel or Serial
    of << "signalTime,rep,mode,time" << std::endl;
    auto signalsVector = createSignalsVector();
    auto gpuResults = runGPUExperiments(signalsVector);
    auto cpuResults = runCPUExperiments(signalsVector);

    for(auto result : gpuResults)
        of << result.first.signalTime << ","
           << result.first.rep << ",Parallel,"
           << result.second << std::endl;
    for(auto result : cpuResults)
        of << result.first.signalTime << ","
           << result.first.rep << ",Serial,"
           << result.second << std::endl;
    return 0;
}
std::vector<SignalAndSS> createSignalsVector() {
    std::vector<SignalAndSS> result;
    for (auto signalTime : signalTimeVector) {
        /* A sinewave signal is created by the use of
         * SynthSignal library, length of the signal
         * is defined by signalTime and this is done
         * as many times as replications' number.
         */
        SynthSignal::Signal signalModel;
        SynthSignal::Interpolation interpolation({}, SynthSignal::InterpolationType::LINEAR);
        interpolation.addPoint(0.0f, 1.0f);
        interpolation.addPoint(signalTime, 1.0f);
        SynthSignal::Interpolation frequencyVariation({}, SynthSignal::InterpolationType::LINEAR);
        frequencyVariation.addPoint(0.0f, 1.0f);
        frequencyVariation.addPoint(signalTime, 1.0f);
        auto wf = new SynthSignal::SineWaveform(
                a,
                omega,
                phi,
                frequencyVariation
        );
        signalModel.addEvent(wf, interpolation);
        auto signal = signalModel.gen(signalTime, sps);
        const float sumOfSquares =
                std::accumulate(signal->begin(), signal->end(), 0.0f,
                                [](float accum, float val) -> float {
                                    accum = accum + val * val;
                                });
        result.push_back(std::make_pair(signal, sumOfSquares));
    }
    return result;
}
/*
 * runGPUExperiments run all experiments
 * under one GPU device.
 */
std::vector<ResultPair> runGPUExperiments(
        const std::vector<SignalAndSS> &signalsAndSSVector) {
    std::vector<ResultPair> results;
    // Initialize device environment
    auto device = bc::system::default_device();
    auto ctx = bc::context(device);
    auto queue = bc::command_queue(ctx, device);
    // Create kernel
    auto program =
            bc::program::create_with_source(sinewaveFitSourceParallel, ctx);
    try {
        program.build();
    } catch(std::exception &e) {
        std::cout << e.what() << std::endl;
        exit(1);
    }
    auto kernel =
        bc::kernel(program,"calculateScoresOfPopulation");
    // Create population related vectors needed in device
    auto dX = bc::vector<bc::float2_>(config.popSize, ctx);
    auto dScores = bc::vector<float>(config.popSize, ctx);
    // Set default kernel parameters
    kernel.set_arg(1,dX);
    kernel.set_arg(2,dScores);
    kernel.set_arg(3,(unsigned int)dX.size());
    kernel.set_arg(6,omegaMin);
    kernel.set_arg(7,omegaMax);
    kernel.set_arg(8,phiMax);
    kernel.set_arg(9,(unsigned int)sps);
    kernel.set_arg(10,a);
    for(auto &signalAndSS : signalsAndSSVector) {
        auto signal = signalAndSS.first;
        auto sumOfSquares = signalAndSS.second;
        // Copy signal to device
        auto dSignal = bc::vector<float>(signal->begin(),signal->end(),queue);
        // Set signal related kernel args
        kernel.set_arg(0,dSignal);
        kernel.set_arg(4,(unsigned int)dSignal.size());
        kernel.set_arg(5,sumOfSquares);

        auto evaluatePopulationLambda =
                [&queue, &kernel, &dSignal, &dScores, &dX, &sumOfSquares](
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
                    bc::copy(hPopulation.begin(),hPopulation.end(),dX.begin(),queue);
                    //Once population has been copied to the device
                    //a parallel calculation of scores must be done
                    queue.enqueue_1d_range_kernel(kernel,0,hPopulation.size(),0);
                    bc::copy(dScores.begin(),dScores.end(),scores.begin(),queue);
                };
        MinusDarwin::Solver solver(config,evaluatePopulationLambda);
        for(size_t rep = 0; rep < replications; rep++) {
            /* Each replication is computed and result is
             * appended to the results vector
             */
            ResultPair result;
            result.first.signalTime = (float)signal->size()/(float)sps;
            result.first.rep = rep;
            auto startTime = boost::chrono::high_resolution_clock::now();
            solver.run(false);
            auto endTime = boost::chrono::high_resolution_clock::now();
            auto duration =
                boost::chrono::duration_cast<boost::chrono::milliseconds>(endTime-startTime);
            result.second = duration.count();
            std::cout << "GPU Time: " << result.second << std::endl;
            results.push_back(result);
        }
    }
    return results;
}
void runCPUCoreExperiment(SignalAndSS signalAndSS,
                          std::vector<ResultPair> *results) {
    auto signal = signalAndSS.first;
    auto sumOfSquares = signalAndSS.second;
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
    MinusDarwin::Solver solver(config,evaluatePopulationLambdaSerial);
    for(size_t rep = 0; rep < replications; rep++) {
        /* Each replication is computed and result is
         * appended to the results vector
         */
        ResultPair result;
        result.first.signalTime = (float)signal->size()/(float)sps;
        result.first.rep = rep;
        auto startTime = boost::chrono::high_resolution_clock::now();
        solver.run(false);
        auto endTime = boost::chrono::high_resolution_clock::now();
        auto duration =
                boost::chrono::duration_cast<boost::chrono::milliseconds>(endTime-startTime);
        result.second = duration.count();
        std::cout << "CPU Time: " << result.second << std::endl;
        results->push_back(result);
    }
}
/* This function will run on async threads
 * where each thread will run a number of
 * replications
 */
std::vector<ResultPair> runCPUExperiments(
        const std::vector<SignalAndSS> &signalsAndSSVector) {
    auto cores = std::thread::hardware_concurrency();
    std::vector<std::vector<ResultPair> > results(
            signalsAndSSVector.size()
    );
    std::vector<std::future<void> > futures;
    size_t count = 0;
    for(size_t core = 0; core<cores; core++) {
        if(count >= signalsAndSSVector.size()) break;
        futures.push_back(std::async(
                std::launch::async,
                runCPUCoreExperiment,
                signalsAndSSVector.at(count),
                &results.at(count)));
        count++;
    }
    while(count < signalsAndSSVector.size()) {
        for(auto &future : futures)
        {
            if(future.wait_for(std::chrono::seconds(0)) ==
                    std::future_status::ready) {
                future = std::async(
                        std::launch::async,
                        runCPUCoreExperiment,
                        signalsAndSSVector.at(count),
                        &results.at(count));
                count++;
                break;
            }
        }
    }
    for(auto &future : futures)
        future.wait();
    auto result = std::vector<ResultPair>();
    for(auto &r : results) {
        for(auto &v : r)
            result.push_back(v);
    }
    return result;
}
