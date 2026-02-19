// Audio player developed from the 1-2: Test Hardware example available to Teensy, with specific intention of playing signals used in my CEAIVS8 project
// Current code is meant to function with just the audio shield attached.

#include <Audio.h>
#include <Wire.h>
#include <SD.h>
#include <SPI.h>
#include <SerialFlash.h>

// General audio preparation
AudioOutputI2S i2s1;
AudioControlSGTL5000 sgtl5000_1;

// Mixer to enable all signals being "active" at once.
AudioMixer4 mixer;                              // Has 4 channels to mix on
AudioConnection patchCord5(mixer, 0, i2s1, 0);  // Mixer to left output
AudioConnection patchCord6(mixer, 0, i2s1, 1);  // Mixer to right output

// When MLS is generated
AudioPlayQueue MLSSignal;
AudioConnection patchCord1(MLSSignal, 0, mixer, 0);  // Sends the MLS to channel 0 in the mixer

// When pure sine is generated
AudioSynthWaveform sineWave;                        // Utilizes pre-made sine wave generator
AudioConnection patchCord2(sineWave, 0, mixer, 1);  // Sends the pure sine to channel 1 in the mixer

// When white noise is generated
AudioSynthNoiseWhite whiteNoise;                      // Utilizes pre-made white noise generator
AudioConnection patchCord3(whiteNoise, 0, mixer, 2);  // Sends the white noise to channel 2 in the mixer

// When white noise is generated
//AudioSynthNoisePink pinkNoise;                       // Utilizes pre-made white noise generator
//AudioConnection patchCord7(pinkNoise, 0, mixer, 2);  // Sends the white noise to channel 2 in the mixer

// When sine sweep is generated
AudioSynthToneSweep sineSweep;                       // Utilizes pre-made sine sweep generator
AudioConnection patchCord4(sineSweep, 0, mixer, 3);  // Sends the sine sweep to channel 3 in the mixer

// Global variables
uint16_t count = 1;
const uint8_t ledPin = 13;        // Pin 13 is the builtin LED
uint32_t LFSRBits = 16;           // Change this between 2 and 32
const unsigned long delayUS = 1;  // Delay in microseconds between bits
uint32_t LFSR;
uint32_t mask;
float sineSweepTime = 0.743;  // Practically decides how fast the frequency is changed in the sine sweep, by saying deltaF/time

void setup() {
  AudioMemory(100);               // Haven't quite figures out yet
  Serial.begin(115200);           // Sets BAUD rate
  sgtl5000_1.enable();            // Turns on audio shield
  sgtl5000_1.volume(1);        // Sets the volume on the audio shield
  sineWave.begin(WAVEFORM_SINE);  // Designates a pure sinusoid
  pinMode(ledPin, OUTPUT);        // Enables the builtin LED
  delay(1000);

  // Set gain for different channels from 0-1
  mixer.gain(0, 0.11);  // 0.0653);  // Gain for MLS
  mixer.gain(1, 0.11);  //0.0793);  // Gain for pure sine
  mixer.gain(2, 0.11);  // 0.105);   // Gain for white noise
  mixer.gain(3, 0.11);  // 0.078);   // Gain for sine sweep
}

void loop() {

  Serial.print("Iteration #");
  Serial.println(count);
  count++;
  sineWave.amplitude(1);

  for (int f = 7; f <= 33; f++) {  // Increments frequency by 1/3 octave bands
    Serial.print("Sine at ");
    Serial.print(octaveFrequency(f));
    Serial.println(" Hz");
    mixer.gain(1, 0.11);  // 0.0793);  // Gain for pure sine
    //sineWave.amplitude(1);                   // Allows further volume control, apart from the gain set earlier
    sineWave.frequency(octaveFrequency(f));  // Sets the current frequency
    wait(1000);
    mixer.gain(1, 0.00005);  // 0.0793);  // Gain for pure sine
    //sineWave.amplitude(0);
    wait(5000);
  }
  //sineWave.frequency(400);
  //wait(5000);
  wait(500);

  Serial.println("White noise");
  whiteNoise.amplitude(1);
  wait(1000);
  whiteNoise.amplitude(0.0);
  wait(5000);
  /*
  Serial.println("Pink noise");
  pinkNoise.amplitude(1);
  wait(1000);
  pinkNoise.amplitude(0.0);
  wait(5000);
*/
  Serial.println("Sine sweep");
  sineSweep.play(1, 50, 20000, sineSweepTime);
  wait(sineSweepTime * 1000);
  wait(5000);

  generateMLS();
  /*
  if (LFSRBits < 32) {  // Looping through various lengths to listen to different MLS's
    LFSRBits++;
  } else if (LFSRBits >= 32) {
    LFSRBits = 2;
    return;
  }*/
  Serial.println("Done");
  delay(5000);
  Serial.println("Disconnect");
  delay(5000);
}

void wait(unsigned int milliseconds) {
  elapsedMillis msec = 0;

  while (msec <= milliseconds) {  // In here, anything can be done. This is just a smart version of the delay function
    digitalWrite(ledPin, HIGH);   // set the LED on
    delay(10);                    // wait for a second
    digitalWrite(ledPin, LOW);    // set the LED off
    delay(20);
  }
}

// Feedback tap map for various left-shifting LFSR lengths (primitive polynomials) with correct taps
uint32_t feedbackTaps(uint8_t bits) {
  switch (bits) {
    case 2: return (1 << 1) | (1 << 0);                             // x^2 + x + 1
    case 3: return (1 << 2) | (1 << 0);                             // x^3 + x + 1
    case 4: return (1 << 3) | (1 << 0);                             // x^4 + x + 1
    case 5: return (1 << 4) | (1 << 2);                             // x^5 + x^3 + 1
    case 6: return (1 << 5) | (1 << 4);                             // x^6 + x^5 + 1
    case 7: return (1 << 6) | (1 << 5);                             // x^7 + x^6 + 1
    case 8: return (1 << 7) | (1 << 5) | (1 << 4) | (1 << 3);       // x^8 + x^6 + x^5 + x^4 + 1
    case 9: return (1 << 8) | (1 << 4);                             // x^9 + x^5 + 1
    case 10: return (1 << 9) | (1 << 6);                            // x^10 + x^7 + 1
    case 11: return (1 << 10) | (1 << 8);                           // x^11 + x^9 + 1
    case 12: return (1 << 11) | (1 << 5) | (1 << 3) | (1 << 0);     // x^12 + x^6 + x^4 + x + 1
    case 13: return (1 << 12) | (1 << 3) | (1 << 2) | (1 << 0);     // x^13 + x^4 + x^3 + x + 1
    case 14: return (1 << 13) | (1 << 12) | (1 << 11) | (1 << 1);   // x^14 + x^13 + x^3 + x 1
    case 15: return (1 << 14) | (1 << 13);                          // x^15 + x^14 + 1
    case 16: return (1 << 15) | (1 << 13) | (1 << 12) | (1 << 10);  // x^16 + x^14 + x^13 + x^11 + 1
    case 17: return (1 << 16) | (1 << 13);                          // x^17 + x^14 + 1
    case 18: return (1 << 17) | (1 << 10);                          // x^18 + x^11 + 1
    case 19: return (1 << 18) | (1 << 17) | (1 << 16) | (1 << 13);  // x^19 + x^18 + x^16 + x^14 + 1
    case 20: return (1 << 19) | (1 << 16);                          // x^20 + x^17 + 1
    case 21: return (1 << 20) | (1 << 18);                          // x^21 + x^19 + 1
    case 22: return (1 << 21) | (1 << 20);                          // x^22 + x^21 + 1
    case 23: return (1 << 22) | (1 << 17);                          // x^23 + x^18 + 1
    case 24: return (1 << 23) | (1 << 22) | (1 << 21) | (1 << 16);  // x^24 + x^23 + x^22 + x^17 + 1
    case 25: return (1 << 24) | (1 << 21);                          // x^25 + x^22 + 1
    case 26: return (1 << 25) | (1 << 5) | (1 << 1) | (1 << 0);     // x^26 + x^6 + x^2 + x 1
    case 27: return (1 << 26) | (1 << 4) | (1 << 1) | (1 << 0);     // x^27 + x^5 + x^2 + x + 1
    case 28: return (1 << 27) | (1 << 24);                          // x^28 + x^25 + 1
    case 29: return (1 << 28) | (1 << 26);                          // x^29 + x^27 + 1
    case 30: return (1 << 29) | (1 << 5) | (1 << 3) | (1 << 0);     // x^30 + x^6 + x^4 + x + 1
    case 31: return (1 << 30) | (1 << 27);                          // x^31 + x^28 + 1
    case 32: return (1 << 31) | (1 << 21) | (1 << 1) | (1 << 0);    // x^32 + x^22 + x^2 + x + 1
    default:
      Serial.println("Unsupported bit length!");
      return 0;
  }
}

void playMLSBit(bool bit, int samplesPerBit = 1, int amplitude = 28000) {  // Bool is the actual input parameter of the function (0 or 1), while samplesPerBit and amplitude are static. Amplitude is set at 28000 to not risk clipping the signal on a speaker
  static int16_t buffer[128];
  static int bufferIndex = 0;

  int16_t value = bit ? amplitude : -amplitude;  // Ternary operator deciding to set the amplitude at + or - the given amplitude based on boolean value

  for (int i = 0; i < samplesPerBit; i++) {
    buffer[bufferIndex++] = value;

    if (bufferIndex == 128) {                                 // Buffer set to 128 bits, conforming to I2S standards
      memcpy(MLSSignal.getBuffer(), buffer, sizeof(buffer));  // Copies buffer data to the AudioPlayQueue, enabling the Teensy to play the MLS
      MLSSignal.playBuffer();                                 // Proprietary play function for the AudioPlayQueue
      bufferIndex = 0;                                        // Resets the bufferindex, so a new sequence can be initialized
    }
  }
}

void generateMLS() {
  if (LFSRBits < 2 || LFSRBits > 32) {  // Double check to ensure 32 bit values are the largest used
    Serial.println("LFSRBits must be between 2 and 32.");
    while (1)
      ;
  }

  mask = (1UL << LFSRBits) - 1;            // Creates a seed containing all 1's
  LFSR = mask;                             // Start with all ones
  uint32_t taps = feedbackTaps(LFSRBits);  // Correct taps for left shift

  Serial.print("Generating MLS with ");
  Serial.print(LFSRBits);
  Serial.println(" bits:");
  Serial.print("The MLS should be ");
  Serial.print((1UL << LFSRBits) - 1);
  Serial.println(" bits long.");

  uint32_t startMLSTime = micros();            // Timer start to monitor execution time
  uint32_t MLSLength = (1UL << LFSRBits) - 1;  // Sets the length to match 2^N-1

  for (uint32_t i = 0; i < MLSLength; i++) {
    bool feedback = __builtin_parity(LFSR & taps);  // Parity of taps using builtin parity to speed up the process
    // Output the feedback bit, NOT the LFSR MSB
    //Serial.print(feedback ? 1 : 0);
    playMLSBit(feedback);

    LFSR <<= 1;  // Shift left
    if (feedback) {
      LFSR |= 1;  // Insert feedback at LSB
    }
    LFSR &= mask;  // Mask to keep LFSRBits width
  }

  uint32_t endMLSTime = micros();  // Timer end to monitor execution time
  Serial.println("\nMLS generation complete.");
  Serial.print("It has taken ");
  Serial.print(endMLSTime - startMLSTime);  // Total execution time given in serial monitor
  Serial.println(" microseconds to calculate and print.");
}

uint16_t octaveFrequency(uint16_t octaveBand) {
  switch (octaveBand) {
    case 1: return 13;
    case 2: return 16;
    case 3: return 20;
    case 4: return 25;
    case 5: return 32;
    case 6: return 40;
    case 7: return 50;
    case 8: return 63;
    case 9: return 80;
    case 10: return 100;
    case 11: return 125;
    case 12: return 160;
    case 13: return 200;
    case 14: return 250;
    case 15: return 315;
    case 16: return 400;
    case 17: return 500;
    case 18: return 630;
    case 19: return 800;
    case 20: return 1000;
    case 21: return 1250;
    case 22: return 1600;
    case 23: return 2000;
    case 24: return 2500;
    case 25: return 3150;
    case 26: return 4000;
    case 27: return 5000;
    case 28: return 6300;
    case 29: return 8000;
    case 30: return 10000;
    case 31: return 12500;
    case 32: return 16000;
    case 33: return 20000;
    default:
      Serial.println("Unsupported octave band!");
      return 0;
  }
}
