### Drug-name misses (100 of 132)

- other_or_devanagari: 67
- latin_misspelling: 33

### Dose misses (64 of 132)

- number_ok_unit_differs: 41
- number_wrong_or_missing: 23

### Examples

- **drug 'Metformin' [latin_misspelling]**
  - REF: Metformin 500 mg सुबह और शाम खाने के बाद लीजिए
  - HYP: mitformin 500 mgs सुबह और शाम खाने के बाद लीजिए
- **dose '500 mg' [number_ok_unit_differs]**
  - REF: Metformin 500 mg सुबह और शाम खाने के बाद लीजिए
  - HYP: mitformin 500 mgs सुबह और शाम खाने के बाद लीजिए
- **drug 'Amoxicillin' [other_or_devanagari]**
  - REF: Amoxicillin 500 mg हर आठ घंटे में पाँच दिन तक लेनी है
  - HYP: amoxycelein 500 mg हर 8 घंटे में 5 दिन तक लेनी है
- **drug 'Azithromycin' [other_or_devanagari]**
  - REF: Azithromycin 250 mg रोज़ एक गोली तीन दिन तक लीजिए
  - HYP: आजित्रोमाइसिन थाई 100 मिल्लीग्राम रोज़ एक गोली तीन दिन तक लीजिए
- **dose '250 mg' [number_wrong_or_missing]**
  - REF: Azithromycin 250 mg रोज़ एक गोली तीन दिन तक लीजिए
  - HYP: आजित्रोमाइसिन थाई 100 मिल्लीग्राम रोज़ एक गोली तीन दिन तक लीजिए
- **drug 'Pantoprazole' [other_or_devanagari]**
  - REF: Pantoprazole 40 mg सुबह खाली पेट लीजिए
  - HYP: pan 2 प्रजोल 40 मिल्ली ग्राम सुबह खाली पेट लीजिए
- **drug 'Cetirizine' [other_or_devanagari]**
  - REF: Cetirizine 10 mg रात को सोने से पहले लीजिए
  - HYP: satyrazine 10 mg रात को सोने से पहले लीजिए
- **dose '10 units' [number_ok_unit_differs]**
  - REF: Insulin 10 units सुबह और 8 units रात को लगाइए
  - HYP: insulin 10 यूनित सुभा और 8 यूनित श्रात को लगाए
- **drug 'Ibuprofen' [latin_misspelling]**
  - REF: Ibuprofen 400 mg दर्द होने पर ही लीजिए
  - HYP: ibu profan 400 mg दर्द होने पर ही लीजिए
