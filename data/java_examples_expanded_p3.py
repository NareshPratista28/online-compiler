# Tahap 3: Topik 9-12 (40 Soal Java Bervariasi)

EXPANDED_JAVA_EXAMPLES_P3 = [
    # ================= TOPIC 9: Array 1 (Basic Arrays) =================
    {
        "topic": "Array 1",
        "studi_kasus": "Sistem inventaris toko sepatu Sneakers (Footwear). Terdapat rak gudang untuk 5 merek sepatu lari terdaftar. Staf toko hendak melihat seluruh rentetan urutan merk sepatu dari rak 1 sampai akhir array sepatu.",
        "tugas": "- Inisialisasi daftar Merek Array bertipe String\n- String merek list sepatu olahraga.\n- Iterasi elemen isi nama dari daftar indeks.",
        "code": """
public class InventarisSepatuLari {
    public static void main(String[] args) {
        String[] daftarMerek = {"Nike", "Adidas", "Puma", "Reebok", "NewBalance"};
        System.out.println("Cetak Database Sepatu Rak Gudang Bawah: ");
        
        for (int m_Merek = 0; m_Merek < ...; m_Merek++) {
            System.out.println("Barang Loker #" + m_Merek + " : " + ...); 
        }
    }
}"""
    },
    {
        "topic": "Array 1",
        "studi_kasus": "Perangkat kasir otomatis memindai Harga produk troli supermarket (Array of Integer). Total keranjang belanja harga barang diakumulasi secara linier menambah struk satu demi satu iterasi hitung loop uang tagihan array tagihan total.",
        "tugas": "- Larik harga list Integer isi nominal belanja\n- Tambahkan harga loop keranjang",
        "code": """
public class TroliSupermarket {
    public static void main(String[] args) {
        int[] keranjangBelanja = {25000, 15000, 8500, 120000, 45000};
        int totalKasir = 0;
        
        System.out.println("Menscan Barcode Barang Klien:");
        for (int scanBiaya = 0; scanBiaya < ...; scanBiaya++) {
            System.out.println("Tit.. Harga Rp " + keranjangBelanja[scanBiaya]);
            totalKasir += ...;
        }
        System.out.println("Bayar di Meja Tagihan Struk Akhir: Rp " + totalKasir);
    }
}"""
    },
    {
        "topic": "Array 1",
        "studi_kasus": "Sistem pengereman ECU (Engine Control Unit) mendeteksi kecepatan sensor 4 Roda ban mobil sekaligus (Kiri-Depan, Kiri-Belakang, Kanan-Depan, Kanan-Belakang). Kecepatan roda harus dicari nilai maksimum tertingginya (Deteksi roda slip spin liar).",
        "tugas": "- Inisialisasi Array putaran Rpm Roda ban List Data 4 indeks sisi\n- Cari angka putaran maksimum array algoritma roda.",
        "code": """
public class SensorRodaSlip {
    public static void main(String[] args) {
        int[] sensorRpmKakiRoda = {820, 815, 825, 1400}; // Roda no.3 liar licin air
        int putaranTertinggi = sensorRpmKakiRoda[0];
        
        for (int sensorPutar = 1; sensorPutar < ...; sensorPutar++) {
            if (... > putaranTertinggi) {
                putaranTertinggi = ...;
            }
        }
        System.out.println("PERINGATAN! Roda slip terdeteksi di RPM Liar: " + putaranTertinggi);
    }
}"""
    },
    {
        "topic": "Array 1",
        "studi_kasus": "Log absen absensi mesin Fingerprint Perusahaan Harian Karyawan (Senin = Hadir, Selasa = Izin, Rabu = Hadir). Data Boolean disatukan array untuk diubah rekap HRD mingguan masuknya absen record data presensi karyawan cuti sakit bolos.",
        "tugas": "- Boolean log Absen Daftar String mingguan bool iter\n- Cetak kehadiran absen daftar laporan true absen false.",
        "code": """
public class RekapFingerprintHRD {
    public static void main(String[] args) {
        boolean[] absenSemingguHrdKerja = {true, true, false, true, true};
        String[] namaHariKerja = {"Sen", "Sel", "Rab", "Kam", "Jum"};
        
        System.out.println("Rekam Laporan Sidik Jari Karyawan Mingguan:");
        for (int hrdHari = 0; hrdHari < ...; hrdHari++) {
            String statusTeks = (...) ? "Hadir" : "Alpa/Izin";
            System.out.println("Hari " + namaHariKerja[hrdHari] + " Status: " + statusTeks);
        }
    }
}"""
    },
    {
        "topic": "Array 1",
        "studi_kasus": "Dalam pemrosesan audio Equalizer (EQ) frekuensi pita 5-Band studio vokal. Suara Bass hingga Treble direkam dalam Array Gain (Desibel). Petugas sound engineer mau menaikkan paksa (Boost) +3 desibel pada setiap frekuensi band.",
        "tugas": "- Pita Array desibel Float/double dari audio EQ rekaman studio\n- Naikkan boost EQ band Array",
        "code": """
public class StudioAudioEqualizer {
    public static void main(String[] args) {
        double[] nilaiBandDesibelEQ = {-2.5, 0.0, 1.5, -1.0, 3.2};
        double dorongBoost = 3.0; // Naikkan semua channel rata
        
        System.out.println("Menggeser Fader Audio Amplifier 5-Band: ");
        for (int p_PitaSuara = 0; p_PitaSuara < ...; p_PitaSuara++) {
            nilaiBandDesibelEQ[p_PitaSuara] += ...;
            System.out.println("Ch " + p_PitaSuara + " Final Gain Output: " + nilaiBandDesibelEQ[p_PitaSuara] + " dB");
        }
    }
}"""
    },
    {
        "topic": "Array 1",
        "studi_kasus": "Kumpulan nilai IPK Mahasiswa Fakultas Teknik (Lulusan Terbaik Cumlaude) ditaruh pada memori Array pecahan desimal. Dicetak berurutan mencari nilai Cuma Lulus indeks array kumulatif memori data pangkalan perguruan tinggi angka presisi rata IPK array.",
        "tugas": "- Cetak daftar IPK desimal float array for.",
        "code": """
public class DatabaseYudisiumIPK {
    public static void main(String[] args) {
        double[] nilaiIpkMhsAlumni = {3.75, 3.92, 3.88, 3.40, 3.99};
        
        System.out.println("Membacakan Kategori Kelulusan Istimewa Cumlaude Kehormatan Terbaik: ");
        for (int cekIuranMhs = 0; cekIuranMhs < ...; cekIuranMhs++) {
            System.out.println("Mahasiswa No Induk " + cekIuranMhs + " Meraih Gelar Prestasi Sarjana, Skor: " + ...);
        }
    }
}"""
    },
    {
        "topic": "Array 1",
        "studi_kasus": "Aplikasi pendataan stasiun pengisian kendaraan listrik umum (SPKLU). Sistem menghitung array KWh daya listrik yang disedot mobil ke mobil colok charger parkir hari ini dari baterai besar charger stasiun SPBU listrik daya besar stasiun SPKLU data Kwh cas mobil batre volt.",
        "tugas": "- Larik stasiun jumlah daya kapasitas SPBU iterasi\n- Array",
        "code": """
public class SpkLuCasKeliling {
    public static void main(String[] args) {
        int[] riwayatKwhDiCasListrik = {45, 60, 22, 10, 85, 30};
        int totalStrum = 0;
        
        System.out.println("Menarik Histori Token PLN Kabel Pengisian EV Station Daya:");
        for (int m = 0; m < ...; m++) {
            System.out.println("Mobil " + m + " Menyedot Daya Setrum Kwh = " + riwayatKwhDiCasListrik[m]);
            totalStrum += ...;
        }
        System.out.println("Sisa Kapasitas Gardu Distribusi Induk Total: " + totalStrum + " KV Listrik");
    }
}"""
    },
    {
        "topic": "Array 1",
        "studi_kasus": "Sensor pengukur jarak Ultrasonic (Sonar) navigasi Robot Pemadam Api dipasang melingkar 6 arah mata angin (Depan, Kiri, Kanan, Belakang, Serong dsj). Tentukan arah sensor mana yang paling jauh dari tembok bebas jalan rute lurus navigasi halang rintang sonar mata lurus array map navigasi map array sensor mata angin kompas robot rute halang jalan rintang aman tembok.",
        "tugas": "- Banding jarak aman terpanjang maksimum sensor loop",
        "code": """
public class RobotPemadamApiSonarRadar {
    public static void main(String[] args) {
        int[] jangkauanTembokCmBot = {15, 30, 2, 180, 50, 45}; // 180 cm di sensor index 3 = bebas lurus
        int spasiAmanJalan = jangkauanTembokCmBot[0];
        int indeksArahBebas = 0;
        
        for (int k = 1; k < ...; k++) {
            if (... > spasiAmanJalan) {
                spasiAmanJalan = ...;
                indeksArahBebas = k;
            }
        }
        System.out.println("Rute teraman tanpa rintangan Tembok Cor di sensor nomor " + indeksArahBebas + " (Berjarak " + spasiAmanJalan + " Cm). Belok Stir Maju Lurus!");
    }
}"""
    },
    {
        "topic": "Array 1",
        "studi_kasus": "Sandi pesan Telegram rahasia masa perang tersusun dari larik array Karakter Huruf Terenkripsi ejaan geser acak karakter array blok. Sistem mendekode cetak gabung pesan menyusun kalimat intel array.",
        "tugas": "- Char array\n- Print tanpa ln gabung huruf array utuh",
        "code": """
public class IntelMessageSandiDekompres {
    public static void main(String[] args) {
        char[] kodeTelegramMorse = {'S', 'E', 'R', 'A', 'N', 'G', 'B', 'E', 'S', 'O', 'K'};
        
        System.out.print("Mendekode Sandi Telepon Intelijen Blok Sekutu Perang Mesin Enigma: ");
        for (int dekripsiHurufH = 0; dekripsiHurufH < ...; dekripsiHurufH++) {
            System.out.print(...);
        }
        System.out.println();
    }
}"""
    },
    {
        "topic": "Array 1",
        "studi_kasus": "Harga cryptocurrency Bitcoin Harian 7 Hari sepekan terakhir volatile terekam dalam vektor memori Array (Tren pasar grafik). Tampilkan grafik harga list hari kripto fluktuasi indeks bursa saham btc pasar uang coin fluktuasi harian hari.",
        "tugas": "- Tampilkan nilai kripto array list loop index",
        "code": """
public class CryptoBursa {
    public static void main(String[] args) {
        int[] btcHargaDolarPerHari = {65000, 62000, 60000, 67000, 71000, 69000, 72000};
        
        System.out.println("Bagan Perdagangan Bitcoin 7 Hari Terakhir: ");
        for (int hariD = 0; hariD < ...; hariD++) {
            System.out.println("Hari ke-" + (hariD + 1) + " Harga Tutup Harian Lilin: $" + ...);
        }
    }
}"""
    },

    # ================= TOPIC 10: Array Multidimensi =================
    {
        "topic": "Array Multidimensi",
        "studi_kasus": "Peta area parkir mall basement (Baris dan Slot). Area Parkir P1 berbentuk Matriks 3 Lantai x 4 Slot. 0=Kosong, 1=Terisi Mobil. Satpam security layar monitor mencari kekosongan cetak map matriks monitor.",
        "tugas": "- Cetak Array multi ukuran 3x4 layout parkiran.",
        "code": """
public class SmartParkingBasementMall {
    public static void main(String[] args) {
        int[][] mapParkiranBasementP = {
            {1, 1, 0, 1},
            {0, 0, 1, 1},
            {1, 1, 1, 0}
        };
        
        System.out.println("Layar Satpam Peta Denah Slot Parkir Mobil Kosong Mall:");
        for (int lantaiP = 0; lantaiP < ...; lantaiP++) {
            System.out.print("Level B" + (lantaiP+1) + " : ");
            for (int slotBlokC = 0; slotBlokC < ...; slotBlokC++) {
                String kondisiKosong = (mapParkiranBasementP[lantaiP][slotBlokC] == 0) ? "[KOSONG]" : "[TERISI]";
                System.out.print(kondisiKosong + " ");
            }
            System.out.println();
        }
    }
}"""
    },
    {
        "topic": "Array Multidimensi",
        "studi_kasus": "Nilai rapor kelas (Matriks Murid x Mata Pelajaran). Ada 3 anak didik, masing masing memegang nilai UN Matematika, IPA, dan Bahasa matrix nilai data nilai siswa 3x3 array sekolahan nilai cetak matriks.",
        "tugas": "- Matriks 2 dimensi for",
        "code": """
public class RaporSiswaSekolah {
    public static void main(String[] args) {
        double[][] nilaiRaporUNMatrikX = {
            {85.5, 90.0, 78.0}, // Siswa 1
            {92.0, 88.5, 95.0}, // Siswa 2
            {70.0, 65.0, 80.5}  // Siswa 3
        };
        
        System.out.println("Transkrip Akademik Data Center Nilai UN Siswa Pusat:");
        for (int n_SiswaBarisIndex = 0; n_SiswaBarisIndex < ...; n_SiswaBarisIndex++) {
            System.out.print("Siswa No " + (n_SiswaBarisIndex+1) + " Mengumpulkan Skor Akademik : ");
            for (int pelKolomP = 0; pelKolomP < ...; pelKolomP++) {
                System.out.print( ... + " | " );
            }
            System.out.println();
        }
    }
}"""
    },
    {
        "topic": "Array Multidimensi",
        "studi_kasus": "Layar piksel TV Monitor Resolusi Rendah (2D Array RGB LED hitam putih). Gambaran piksel gambar emoji smiley kotak dot matrix monitor layar piksel 4x4 array kotak warna cetak gambar monitor bit.",
        "tugas": "- Tampilkan titik resolusi display",
        "code": """
public class LayarLcdPikselDot {
    public static void main(String[] args) {
        char[][] layarResolusiPixelMux = {
            {'#', ' ', ' ', '#'},
            {'#', ' ', ' ', '#'},
            {' ', '#', '#', ' '},
            {'#', ' ', ' ', '#'}
        };
        
        System.out.println("Merender VRAM Kartu Grafis Resolusi Jaman Retro Dot Matrix:");
        for (int resY = 0; resY < ...; resY++) {
            for (int resX = 0; resX < ...; resX++) {
                System.out.print(...);
            }
            System.out.println(); // Lompat barisan scanline TV LCD Tembak
        }
    }
}"""
    },
    {
        "topic": "Array Multidimensi",
        "studi_kasus": "Posisi pion catur di atas papan (Koordinat 2D Board). Sebuah matriks 8x8 papan berisi nama singkatan bidak huruf String bidak benteng rajam menteri prajurit pion papan susunan perang papan bidak tempur kayu catur petak pilar pilar kordinat XY letak catur.",
        "tugas": "- Hanya cetak 3x3 sudut pojok benteng catur saja matriks (Sebagian)",
        "code": """
public class PapanCaturKayu {
    public static void main(String[] args) {
        String[][] papanBidangCaturMatrik = {
            {"B", "K", "G"},
            {"P", "P", "P"},
            {"_", "_", "_"}
        };
        
        System.out.println("Benteng Pasukan Kerajaan Sudut Pertahanan:");
        for (int barisCaturY = 0; barisCaturY < ...; barisCaturY++) {
            for (int kolomLangkahX = 0; kolomLangkahX < ...; kolomLangkahX++) {
                System.out.print("[" + ... + "]");
            }
            System.out.println();
        }
    }
}"""
    },
    {
        "topic": "Array Multidimensi",
        "studi_kasus": "Kalender pemesanan kursi pesawat mingguan (7 Hari x 3 Sesi Waktu Pagi Siang Malam). True artinya kursi ludes tiket terjual, false artinya kursi penerbangan bandara sisa lowong tiket maskapai matriks boolean penerbangan sisa kosong jadwal jam kursi matrix kalender array minggu 7x3 seat.",
        "tugas": "- Mencari ketersediaan jadwa kalender baris kolom loop ganda",
        "code": """
public class TicketBandaraBookingArray {
    public static void main(String[] args) {
        boolean[][] jadwalAviasiSemingguSeatX = {
           {true, true, false}, // Senin
           {false, true, true}, // Selasa
           {true, true, true}   // Rabu (Full Beli)
        };
        
        System.out.println("Cek Kursi Pesawat Yang Kosong (False):");
        for (int hariD = 0; hariD < ...; hariD++) {
            for (int sesiJamH = 0; sesiJamH < ...; sesiJamH++) {
                if (!...) {
                    System.out.println("Ditemukan Kursi Kosong (Tiket Sisa) di Hari index " + hariD + " Jam Sesi " + sesiJamH);
                }
            }
        }
    }
}"""
    },
    {
        "topic": "Array Multidimensi",
        "studi_kasus": "Pertanian hidroponik susun rak atas dan jajar laci. 2D array menampung tinggi tanaman bayam matriks air pipa pvc rak ukuran centimeter pipa hidroponik air matrix sayur botani tinggi tumbuh pupuk array air daun ukuran sayur lapis rak tinggi data hijau segar lapis pipa data array centi meter botani.",
        "tugas": "- Cetak nilai centimeter array hidroponik panen",
        "code": """
public class HidroponikPipaBayam {
    public static void main(String[] args) {
        int[][] rakSayurBayamCmTinggi = {
            {12, 14, 15},
            {10, 11, 13},
            {2, 3, 1}
        };
        
        System.out.println("Scan Lidar Tanaman Sayur Hijau Indoor Pipa Lapis Rak:");
        for (int pipaLapisV = 0; pipaLapisV < ...; pipaLapisV++) {
            for (int lubangBorPipaH = 0; lubangBorPipaH < ...; lubangBorPipaH++) {
                System.out.print(rakSayurBayamCmTinggi[pipaLapisV][lubangBorPipaH] + "cm, ");
            }
            System.out.println();
        }
    }
}"""
    },
    {
        "topic": "Array Multidimensi",
        "studi_kasus": "Kalkulasi matrix perkalian tensor bobot Artificial Intelligence (Matriks 2x2 Bobot Neuron Synapse). Bobot otak memori angka koma desimal ganda saraf syaraf otak jaringan node buatan double data matematika jaringan perulangan jala angka.",
        "tugas": "- Cetak Double Jaringan Matrix for",
        "code": """
public class BrainNodeTensorBobot {
    public static void main(String[] args) {
        double[][] bobotSynapseOtakAI = {
            {0.15, -0.44},
            {0.88, 0.02}
        };
        
        System.out.println("Melemparkan Forward Pass Aktivasi Neuron Layer:");
        for (int syarafInp = 0; syarafInp < ...; syarafInp++) {
            for (int syarafOut = 0; syarafOut < ...; syarafOut++) {
                System.out.print("W(" + syarafInp + "," + syarafOut + ")=" + ... + " | ");
            }
            System.out.println();
        }
    }
}"""
    },
    {
        "topic": "Array Multidimensi",
        "studi_kasus": "Lemari brankas laci dokumen gudang kargo arsip pelabuhan kontainer. (Laci x Loker arsip baris koordinat string dokumen laci baris map surat kontainer box alamat resi logistik tumpukan matrix kontainer 2 lapis lokasi alamat rak peti kemas box laci barang tumpukan array matrix barang).",
        "tugas": "- String array 2 dimensi gudang kontainer",
        "code": """
public class ArsipGudangKontainer {
    public static void main(String[] args) {
        String[][] rakPetiKemalArsip = {
            {"Barang_A1", "Barang_A2"},
            {"Barang_B1", "Barang_B2"}
        };
        
        System.out.println("Pencarian Manifest Peti Kemas Barang Tumpukan Baja:");
        for (int tumpukLantaiI = 0; tumpukLantaiI < ...; tumpukLantaiI++) {
            for (int jejerLaciJ = 0; jejerLaciJ < ...; jejerLaciJ++) {
                System.out.println("Derek Robot Pelabuhan Angkat Laci: " + ...);
            }
        }
    }
}"""
    },
    {
        "topic": "Array Multidimensi",
        "studi_kasus": "Sistem curah hujan BMKG bulanan di 3 provinsi. Array matriks 2D Double curah hujan mm intensitas iklim rata rata awan per provinsi kota baris tabel koordinat hari hujan debit stasiun pencatat observatorium data cuaca suhu hujan matrix array tabel kota stasiun provinsi iklim cuaca.",
        "tugas": "- Data cuaca matrix for loop 2 kali",
        "code": """
public class BmkgCurahHujanKawasan {
    public static void main(String[] args) {
        double[][] debitCurahHujanMiliMeterLokasi = {
            {120.5, 140.2}, // Provinsi Jatim
            {90.0, 85.5},   // Provinsi Jateng
            {210.5, 230.1}  // Provinsi Jabar Ciawi Puncak
        };
        
        System.out.println("Rekam Alat Takar Cuaca Awan Observasi Stasiun Regional Hujan Lebat:");
        for (int p_ProvinsiY = 0; p_ProvinsiY < ...; p_ProvinsiY++) {
            for (int d_DerahKotaX = 0; d_DerahKotaX < ...; d_DerahKotaX++) {
                System.out.print("Stasiun " + p_ProvinsiY + "-" + d_DerahKotaX + " Air Debit: " + ... + " mm , ");
            }
            System.out.println();
        }
    }
}"""
    },
    {
        "topic": "Array Multidimensi",
        "studi_kasus": "Bagan jam kerja shift mesin pabrik penenun benang tekstil kain baju jahit (Matriks 3 Shift x 5 Mesin tenun pabrik). Data total kain diproduksi pabrik per shift meter kain matrix panjang kain mesin jahit 2 array tenun.",
        "tugas": "- Jumlahkan semua total nilai dalam matriks for bersarang (Sum Total Array)",
        "code": """
public class PabrikTenunKainShiftJam {
    public static void main(String[] args) {
        int[][] gulungKainMeterGudangDibuat = {
            {100, 105, 90}, // Shift Pagi (3 Mesin)
            {110, 120, 115},// Shift Siang
            {80, 85, 95}    // Shift Malam
        };
        int totalGulunganTerbuatOmzet = 0;
        
        System.out.println("Menghitung Omzet Harian Produksi Garmen Kapas Tenun...");
        for (int shiftPekerjaWaktu = 0; shiftPekerjaWaktu < ...; shiftPekerjaWaktu++) {
            for (int mesinTenunMesin = 0; mesinTenunMesin < ...; mesinTenunMesin++) {
                totalGulunganTerbuatOmzet += ...;
            }
        }
        System.out.println("Kontainer Baju Kain Omzet Siap Jual Pasar Grosir Tanah Abang: " + totalGulunganTerbuatOmzet + " Meter Bal.");
    }
}"""
    },

    # ================= TOPIC 11: Fungsi Static =================
    {
        "topic": "Fungsi Static",
        "studi_kasus": "Kalkulator konversi uang kurs asing (Forex Bank). Fungsi pecahan menukar uang Rupiah Turis Bali ke nilai nominal Dollar AS secara fungsi terpisah modul static modul angka fungsi uang kalkulasi turis devisa penukaran money changer counter uang bank pecahan fungsi terpisah return ganda pemanggilan.",
        "tugas": "- Buat fungsi konversiIbr\n- Return dobel kurs",
        "code": """
public class MoneyChangerKonversiFungsi {
    
    // Fungsi Method konversi nilai
    public static double tukarRupiahKeDolar(double uangRupiahIdrRaw) {
        double kursTetapDolar = 15500.0;
        return uangRupiahIdrRaw / ...;
    }
    
    public static void main(String[] args) {
        double lembaranRupiahTuris = 5000000.0; // Lima juta
        System.out.println("Turis datang menyerahkan Gepokan Koper " + lembaranRupiahTuris + " IDR");
        
        double hasilTukarBeliKertasValas = ...;
        
        System.out.println("Teller Bank Memberi Kembalian Pecahan Asing Tunai: $" + hasilTukarBeliKertasValas);
    }
}"""
    },
    {
        "topic": "Fungsi Static",
        "studi_kasus": "Bengkel perbaikan bodi mobil (Ketok Magic Las). Fungsi kalkulasi harga estimasi asuransi perbaikain lecet panel pintu body. Input adalah sisi panel rusak lecet (Integer) di kalikan harga per panel dempul cat biaya tukang tarif perbaik bodi ketok las suku cadang part modul fungsi return invoice bengkel bayar nota faktur.",
        "tugas": "- Integer fungsi harga kerusakan",
        "code": """
public class BengkelKetokMagicFakturTukang {

    public static int hitungBiayaDempulCatOven(int jumlahPanelPenyokLecet) {
        int tarifOvenPerPanelBodi = 750000;
        return jumlahPanelPenyokLecet * ...;
    }

    public static void main(String[] args) {
        int kerusakanKecelakaanPenyokTotalPanel = 4;
        System.out.println("Mekanik Mesin Mengecek " + kerusakanKecelakaanPenyokTotalPanel + " Sisi Body Tertabrak Rusak.");
        
        int asuransiCairKlaim = ...;
        
        System.out.println("Manager Penilai Kasir Bengkel Mengeluarkan Tagihan: Rp " + asuransiCairKlaim);
    }
}"""
    },
    {
        "topic": "Fungsi Static",
        "studi_kasus": "Restoran masakan cepat saji kalkulator kasir kalori. Fungsi penghitung total asupan kalori burger ayam minuman bersoda kalori saji hidang. Method fungsi makanan porsi makan kalori nutrisi burger pemanggilan modular diet nutrisi fungsi kalori burger ayam jumlah pesanan porsi menu piring berat fungsi hitung void makanan.",
        "tugas": "- Hitung integer kalori ayam return kali saji porsi pesanan pengunjung.",
        "code": """
public class KaloriDietMenuAyamGoreng {

    public static int kalkulasiLemakKaloriAyamFastFood(int porsiKenyangMakanBurgerAyam) {
        int kaloriSatuKemasBurger = 850;
        return porsiKenyangMakanBurgerAyam * ...;
    }

    public static void main(String[] args) {
        int traktirPestaPorsiDipesanKasirTamuSajiLapar = 3;
        
        int kolesterolMasukBadan = ...;
        
        System.out.println("Anak Kost Makan Pesta Gila Menelan " + kolesterolMasukBadan + " Kkal Kalori Jahat Kolesterol Numpuk Perut!");
    }
}"""
    },
    {
        "topic": "Fungsi Static",
        "studi_kasus": "Aplikasi jasa desain grafis brosur menghitung harga bayar desain cetak lembar per ukuran spanduk banner baliho pinggir jalan. Dimensi meter persegi dikali tarif print tinta flexi meteran outdoor mesin cetak banner printing offset fungsi void int double fungsi parameter meter spanduk tarif meteran tinta jalan raya tiang billboard iklan visual grafis desain poster kertas resolusi.",
        "tugas": "- Hitung double kalkulasi luas x harga.",
        "code": """
public class PercetakanBannerOffsetJalan {

    // Return luas tarif billboard
    public static double estimasiNotaCetakSpandukMeteranFlexiKain(double panjangSpandukX, double lebarSpandukY) {
        double hargaCetakMeterPersegiPixel = 25000.0;
        return (panjangSpandukX * lebarSpandukY) * ...;
    }

    public static void main(String[] args) {
        double p = 3.5;
        double l = 2.0;
        System.out.println("Order Spanduk Kampanye Pilkada Pemilu Tiang Ukuran " + p + "x" + l + " M");
        
        double bayarDPPrintingTintaOffset = ...;
        
        System.out.println("Mesin Gulung Printing Tinta Flexi Besar Bergerak, Harap Bayar Di Muka Lunas Rp " + bayarDPPrintingTintaOffset);
    }
}"""
    },
    {
        "topic": "Fungsi Static",
        "studi_kasus": "Klinik gigi cek tarif cabut gigi graham tambalan kalkulasi fungsi bayaran klinik. Dokter bedah pasang bor harga bor cabut fungsi gigi sakit fungsi metode static modular pemanggilan bayaran tindakan bedah gusi per gigi parameter kawat medis bor cabut.",
        "tugas": "- Fungsi biaya rawat cabut",
        "code": """
public class KlinikGigiOrthodontistDokterBedah {

    public static int hitungNotaRujukCabutGigiBiusLokalRahang(int jumlahGigiPatahBerlubang) {
        int tarifBiusSatuGigiRonggaTindakan = 450000;
        return jumlahGigiPatahBerlubang * ...;
    }

    public static void main(String[] args) {
        int pasianNyeriAbsesGigiBusukCabut = 2;
        
        int bayarResepApotekSusterKasirKlinikTindakan = ...;
        
        System.out.println("Kumuh Membersihkan Kapas Darah, Pasien Sakit Meringis Bayar Ke Kasir Depan Suster Total Rp " + bayarResepApotekSusterKasirKlinikTindakan);
    }
}"""
    },
    {
        "topic": "Fungsi Static",
        "studi_kasus": "Aplikasi kebugaran angkat beban Gym kalibrasi dumbell besi piring plat barbell kalibrasi fungsi static perhitungan berat angkat bar plat kg besi konversi pon berat piring plat berat fungsi parameter angkat kalibrasi piring bar lengan beban besi binaraga gym mesin piring kalibrasi beban pound kg fungsi static hitung beban void lengan piring.",
        "tugas": "- Fungsi konversi beban.",
        "code": """
public class GymBesiBarbellFitnessLatihanBeban {

    public static double ubahPoundKeKgPlat(double beratBebanPlastikKaratPound) {
        return beratBebanPlastikKaratPound * ...; // 1 lbs = 0.453592 kg
    }

    public static void main(String[] args) {
        double bebanPlatPoundLbsUSA = 45.0; // Piring besar olimpiade
        
        double konversiAkurasiOlimpiadeGymKg = ...;
        
        System.out.println("Binaragawan Menset Piring Barbel Hitam " + bebanPlatPoundLbsUSA + " lbs setara Beban Seberat " + konversiAkurasiOlimpiadeGymKg + " Kg Timbangan Akurat.");
    }
}"""
    },
    {
        "topic": "Fungsi Static",
        "studi_kasus": "Software rental PS (Playstation) penyewa main game sewa TV billing warnet konter. Durasi jam dikali tarif rental billing pemanggilan fungsi harga operator kasir modul penjaga warnet ps jam sewa lama fungsi rental main tv.",
        "tugas": "- Fungsi hitung billing PC rental",
        "code": """
public class RentalPlaystationBillingWarnetSelep {

    public static int tagihanRentalSewaBillingJamLedTv(int jamBermainJoystickStikSewa) {
        int tarifPsPerJam = 5000;
        return jamBermainJoystickStikSewa * ...;
    }

    public static void main(String[] args) {
        int bocilMainSewaTamatPSJamJenuh = 4;
        
        int tagihanTerkumpulKasirBilingOpWarning = ...;
        
        System.out.println("Peringatan Billing Sisa Waktu 5 Menit Habis Lampu Merah! Bayar Rp " + tagihanTerkumpulKasirBilingOpWarning);
    }
}"""
    },
    {
        "topic": "Fungsi Static",
        "studi_kasus": "Fungsi utilitas String manipulator, mengecek nama pengguna huruf kapital konversi ke huruf kecil pendaftaran form web user kapital pendaftaran kecil string converter manipulasi web aplikasi form parameter daftar string registrasi peladen pengguna string rubah.",
        "tugas": "- Return tipe string huruf kecil (toLowerCase)",
        "code": """
public class FormWebRegistrasiConvertStringAmanKarakter {

    public static String paksaHurufKecilEmailServerDatabaseBersih(String inputEmailHurufAcakUserCaplockRaw) {
        return inputEmailHurufAcakUserCaplockRaw....;
    }

    public static void main(String[] args) {
        String inputAlamakRusakEmail = "SUkamDi@GMAIl.cOM";
        
        String emailTersetrikaRapiStandarInternet = ...;
        
        System.out.println("Berhasil Mendata Ulang Database Index Email Client Huruf Standar: " + emailTersetrikaRapiStandarInternet);
    }
}"""
    },
    {
        "topic": "Fungsi Static",
        "studi_kasus": "Robot pencari mineral emas gali sensor mendeteksi kedalaman tanah galian meter konversi inci fungsi robot tambang meter inci pengukuran satuan deteksi parameter konversi meter ke inci galian emas robot parameter fungsi gali bor surveyor surveyor surveyor pemetaan tanah.",
        "tugas": "- Konversi meter ke inci",
        "code": """
public class SurveyorTambangEmasSensorGaliBilah {

    public static double meterGalianKeInciBorMataEmasUK(double kedalamanTanahHancurMeterKerasBatu) {
        return kedalamanTanahHancurMeterKerasBatu * ...; // 1 m = 39.37 inch
    }

    public static void main(String[] args) {
        double galinBorangTanahMeter = 1.5;
        
        double inchLogamKaratPanjangDeteksi = ...;
        
        System.out.println("Monitor Alat Bor Rantai Mengeluarkan Bunyi Deteksi Emas Cincin Logam Kedalaman Tanah Hitam = " + inchLogamKaratPanjangDeteksi + " Inch Tanah Dalam.");
    }
}"""
    },
    {
        "topic": "Fungsi Static",
        "studi_kasus": "Kalkulasi pajak PPN resto makanan makanan kasir struk pajak fungsi porsi potongan nilai PPN nominal struk tagihan belanja harga menu pajak struk resto hitung struk fungsi param PPN.",
        "tugas": "- Hitung PPN 11%",
        "code": """
public class PajakStrukPpnRestoMakananMejaMakanTagihanTax {

    public static double pungutPajakPpnNasionalKasirResto(double totalNgelahapMakanTamuMeja) {
        return totalNgelahapMakanTamuMeja * ...; // PPN 11% / 0.11
    }

    public static void main(String[] args) {
        double mejaNomorSatuGagapPesanPenuhTagihanBelumPajak = 100000.0;
        
        double setorPajakPpnKeNegaraKasBon = ...;
        double bayarPelangganUtuh = mejaNomorSatuGagapPesanPenuhTagihanBelumPajak + setorPajakPpnKeNegaraKasBon;
        
        System.out.println("Cetak Bon Printer Termal... Makanan = Rp " + mejaNomorSatuGagapPesanPenuhTagihanBelumPajak + " Plus PPN = " + setorPajakPpnKeNegaraKasBon + " Lunas Penuh = Rp " + bayarPelangganUtuh);
    }
}"""
    },

    # ================= TOPIC 12: Fungsi Rekursif =================
    {
        "topic": "Fungsi Rekursif",
        "studi_kasus": "Perakitan kotak boneka Rusia (Matryoshka). Boneka boneka kayu dipecah setengah dibuka isi boneka kecil belah kotak kurangi ukuran kayu boneka sampai tak bisa belah base case rekursif belahan lapisan membelah kotak boneka rekursi boneka rusia lapis mainan pahat boneka buka kotak kayu mainan kayu buatan pahatan rusia layer rekursi lapisan memori tumpukan call stack.",
        "tugas": "- Method rekursif kurang ukuran blok boneka param - 1 base 0",
        "code": """
public class BonekaRusiaRekursiMatryoshkaBongkarUkiranLayer {

    public static void bukaLapisBonekaRusiaDalam(int levelCangkangBesarLuaran) {
        if (levelCangkangBesarLuaran == 0) { // Base Case
            System.out.println("Tadaa! Ditemukan Boneka Inti Kayu Utuh Bayi Terkecil Terakhir Padat. Stop!");
            return;
        }
        System.out.println("Mencabut Paksa Setengah Pundak Cangkang Boneka Ke-" + levelCangkangBesarLuaran + " Keluar Bunyi Kreeeek.");
        // Recursive Call
        ...(levelCangkangBesarLuaran - 1); 
    }

    public static void main(String[] args) {
        System.out.println("Membeli Souvenir Tua Mengerikan Boneka Kayu Berwajah Mengerikan dari Pasar Loak Moscow Tertutup Salju Lapis Kutub...");
        bukaLapisBonekaRusiaDalam(4);
    }
}"""
    },
    {
        "topic": "Fungsi Rekursif",
        "studi_kasus": "Virus replikasi Worm Komputer folder payload root sistem virus copy paste memecah biner injeksi virus panggil diri sendiri infeksinya folder param minus file size infeksi tumpukan file system root c drive virus bahaya cacing trojan replikasi menyarang payload payload injeksi fungsi serang copy paste diri memori folder sistem rekursi panggil diri virus menyebar payload ancaman heker jahat virus botnet kompi jaringan.",
        "tugas": "- Fungsi hack replikasi param depth infeksi kurang",
        "code": """
public class HackerWormVirusPayloadReplikasiFolderrCacingKodeBusukAkarBaseBiner {

    public static void menyebarKeAkarFolderRootOSInjeksiServerFile(int kedalamanStrukturFolderDriveTujuanSembunyi) {
        if (kedalamanStrukturFolderDriveTujuanSembunyi == 0) {
            System.out.println("Kernel Sistem Utama (C:) Hancur Terinfeksi Payload Trojan Payload. Batas Maksimal Akses Drive Akar Ter-Hijack Semua!");
            return;
        }
        System.out.println("Copy File 'system_bug_worm.exe' Ke Sub-Direktori Kedalaman Ke - " + kedalamanStrukturFolderDriveTujuanSembunyi);
        // Memanggil fungsi virus merusaknya diri serang akar terdalam
        ...(kedalamanStrukturFolderDriveTujuanSembunyi - 1);
    }

    public static void main(String[] args) {
        System.out.println("Admin Bodoh Meng-Klik Tautan Judul Undian Palsu Browser Celah Bahaya Keamanan Jaringan Tembus!");
        menyebarKeAkarFolderRootOSInjeksiServerFile(3);
    }
}"""
    },
    {
        "topic": "Fungsi Rekursif",
        "studi_kasus": "Ledakan bom kembang api percabangan mekar ke langit malam. Rekursi percikan serpihan memecah menjadi lebih kecil rekursi kembang ledakan api kecil param daya ledak kembang ledak rekursi panggil percik serpihan kembang api rekursi memanjat serbuk mesiu kembang percik ledakan rekahan letusan mekar api udara memori ekor percikan kecil serbuk mesiu pecah bercabang kembang letusan panggil function fungsi letusan ledakan batas panggill base case mekar.",
        "tugas": "- Fungsi rekursi ledakan kurang partikel.",
        "code": """
public class KembangApiMekarAngkasaSerbukMagnesiumTerangMerconBakarLedakanPartikelTerbangNyalaRoketMalam {

    public static void memecahBungaPartikelSerpihanMerconMagnesiumTerangPusat(int diameterBulatanLedakanTerangSerbukSerpihanPecahKilatCahayaKembang) {
        if (diameterBulatanLedakanTerangSerbukSerpihanPecahKilatCahayaKembang <= 0) {
            System.out.println("Eksplosi Berakhir Gelap Asap Tebal Abu Mesiu Putih Menggumpal Menutupi Bintang Malam Angin Bawa Meniup Ilang Asap.");
            return;
        }
        System.out.println("DUARR! Serpihan Mencabang Meledak Pecah Lagi Menjadi Bunga Ukuran " + diameterBulatanLedakanTerangSerbukSerpihanPecahKilatCahayaKembang + " Centimeter Bunga Api Warna Warni.");
        
        // Letuskan bunga yang berukuran lebih kecil dari partikel bunga ini
        ... (diameterBulatanLedakanTerangSerbukSerpihanPecahKilatCahayaKembang - 2);
    }

    public static void main(String[] args) {
        System.out.println("Menyulut Sumbu Lilin Batang Roket Kardus Mercon Besar Lari Jauuuh... Suussssssttttt Dorong Angkasa Melesat Terang Roket Naik Udara...");
        memecahBungaPartikelSerpihanMerconMagnesiumTerangPusat(5);
    }
}"""
    },
    {
        "topic": "Fungsi Rekursif",
        "studi_kasus": "Menghitung pangkat (Eksponensial) Rekursi nilai bilangan pangkat dasar panggil diri kurang kurang param rekursi matematika dasar pangkat iterasi pemanggilan nilai bilangan tumpuk call memanggil param matematika angka kalikan dasar pengkali hasil kalikan hasil nilai dasar dengan pangkat kurangi rekursi basis rumus power pangkat fungsi return kalikan batas nol base satu pangkat satu basis matematika pangkat.",
        "tugas": "- Fungsi hitung pangkat rekursif param pangkat",
        "code": """
public class MatematikaPangkatPowerBaseExponentEksponensialRekursifKalkulasiDasarAngkaTumpukLoopPengkaliNumBaseMtkPerpangkatanCallRumusAlgoritma {

    public static int hitungPangkatEksponensialBerantaiDasarNumIterBaseRekursif(int dasarHitungKaliNumPangkatPerpangkatanBaseA, int derajatPangkatEksponensialNumpangkatAtas) {
        if (derajatPangkatEksponensialNumpangkatAtas == 0) {
            return 1;
        }
        // Rekursi Eksponen
        return dasarHitungKaliNumPangkatPerpangkatanBaseA * ... (dasarHitungKaliNumPangkatPerpangkatanBaseA, derajatPangkatEksponensialNumpangkatAtas - 1);
    }

    public static void main(String[] args) {
        int dasarKu = 2;
        int pangkatKuu = 4;
        System.out.println("Hitung Kuasa Angka Kuadrat Kubik Tumpuk Kali Lipat Ganda " + dasarKu + " Pangkat ^ " + pangkatKuu);
        
        int outputHasilPowerKaliListNumAkhirGenapNumBiner = ... (dasarKu, pangkatKuu);
        
        System.out.println("Mesin Otak Kalkulator Berhenti Hasil Akhir Adalah = " + outputHasilPowerKaliListNumAkhirGenapNumBiner);
    }
}"""
    },
    {
        "topic": "Fungsi Rekursif",
        "studi_kasus": "Ekspedisi pendakian Gunung berliku rekursi susur basecamp ketinggian panggil diri naiki pos pendakian kurang sisa pos basecamp camp rekursi daki gunung puncak panggil pos sampai nol pos basecamp base puncak sisa pos rekursi basecamp ketinggian panggil.",
        "tugas": "- Susur pos turun param rekursi",
        "code": """
public class RuteGunungBasecampCampsiteMendakiPosSummitRekursiHikingRanselKakiGunungPuncakCampPendakianTracking {

    public static void melangkahkanKakiMendakiKePosPosSelanjutnyaCampsiteCheckPoint(int sisaTargetPosDepanPendakianSummitTrekJalurJauhBatuAkarPohonNanjak) {
        if (sisaTargetPosDepanPendakianSummitTrekJalurJauhBatuAkarPohonNanjak == 0) {
            System.out.println("Alhamdulillah Muncak Berkibar Bendera Udara Tipis Kabut Puncak Abadi Gunung Semburat Fajar Masuk Angin!");
            return;
        }
        System.out.println("Sampai Ke Pos Singgah Ke-" + (5 - sisaTargetPosDepanPendakianSummitTrekJalurJauhBatuAkarPohonNanjak) + ", Luruskan Kaki Kram Istirahat Semenit Seduh Kopi Hitam Lanjut Tarik Nafas Panjang Berjalan Beban Ransel...");
        ...(sisaTargetPosDepanPendakianSummitTrekJalurJauhBatuAkarPohonNanjak - 1);
    }

    public static void main(String[] args) {
        System.out.println("Berangkat Dari Kaki Gerbang Jalur Desa Pendakian Gelap Malam Hutan Angker Kabut Embun Dingin Trek Tanah Basah Beranjak Kaki Naiki Bukit Menuju Pos Cek Jalur Ekspedisi Map Alam Rimba Bawa Tenda Kompor Carier Berat...");
        melangkahkanKakiMendakiKePosPosSelanjutnyaCampsiteCheckPoint(4);
    }
}"""
    },
    {
        "topic": "Fungsi Rekursif",
        "studi_kasus": "Sifat bakteri korosi memakan ketebalan besi lambung kapal karam dasar laut karat rekursi karat gigit ketebalan plat minus inci panggil diri rekursi lubang karat bolong laut dalam korosi rekursif.",
        "tugas": "- Rekursif pengurangan pelat plat",
        "code": """
public class KorosiBesiKapalKaramLautanGaramGanasKaratAsamBakteriBolongPelatBajaRonggaLubangRuntuhRekursiDasarSamudra {

    public static void digerogotiKaratGaramLautanAsamMenjadiSerbukAir(int sisaKetebalanInciBesiBajaPlatKaramLambungRongsokanKapalTitanicKaramGelapAirArus) {
        if (sisaKetebalanInciBesiBajaPlatKaramLambungRongsokanKapalTitanicKaramGelapAirArus <= 0) {
            System.out.println("TENGGELAM RUNTUH! Diding Lambung Pecah Berkeping Jadi Debu Pasir Besi Larut Arus Dasar Laut Angin Biru Gelap Hancur Tulang Paus Hidup Kepiting Ikan Buta Merayap Reruntuhan Tenggelam.");
            return;
        }
        System.out.println("Kropos Mengelotok Karat Terkikis Garam Laut Ketebalan Besi Sisa Menganga Mengkriuk " + sisaKetebalanInciBesiBajaPlatKaramLambungRongsokanKapalTitanicKaramGelapAirArus + " Menipis Inci...");
        ... (sisaKetebalanInciBesiBajaPlatKaramLambungRongsokanKapalTitanicKaramGelapAirArus - 1);
    }

    public static void main(String[] args) {
        System.out.println("Mendeteksi Titik Karam Koordinat Sonar Satelit Ekspedisi Puing Puing Karam Tersembunyi Palung Mariana Tekanan Air Gelap Dingin Mencekam Dasar Pasir Kapal Tempur Karam Jatuh Terhantam Karang Badai Perang...");
        digerogotiKaratGaramLautanAsamMenjadiSerbukAir(4);
    }
}"""
    },
    {
        "topic": "Fungsi Rekursif",
        "studi_kasus": "Kalkulator pembagian rekursif cara pegurangan memotong bilangan dibagi kurangi panggil fungsi hasil nambah 1 pembagi bilangan rekursi dasar potong pengurangan panjang berurut bagi fungsi rekursi sisa per panggil panggil.",
        "tugas": "- Pembagian dengan cara rekursi pemotongan kurang loop parameternya basis tambah 1 return.",
        "code": """
public class PembagianRekursifAlgoritmaMatematikaGila {

    public static int bagiCaraSengsaraRekursifPotong(int pembilangAtasBesarNominalBagi, int penyebutBawahAngkaPemotongLebihKecilBagi) {
        if (pembilangAtasBesarNominalBagi < penyebutBawahAngkaPemotongLebihKecilBagi) {
            return 0; // Sisa bagi gak bisa dipotong penuh
        }
        return 1 + ... (pembilangAtasBesarNominalBagi - penyebutBawahAngkaPemotongLebihKecilBagi, penyebutBawahAngkaPemotongLebihKecilBagi);
    }

    public static void main(String[] args) {
        System.out.println("Menyetel Gigi Abacus Kalkulator Mesin Hitung Tua Pemotong Angka: 20 per 5 Bagi Potong...");
        int hasilBagiBeratGilaCPU = ... (20, 5);
        System.out.println("Roda Kalkulator Stop. Hasil Pemotongan Utuh Penuh Pecah Bagi Bagian Angka Dasar Genap Adalah = " + hasilBagiBeratGilaCPU);
    }
}"""
    },
    {
        "topic": "Fungsi Rekursif",
        "studi_kasus": "Simulasi Pantulan bola bekel karet. Fungsi memanggil diri jatuh pantul ketinggian kurangi setengah bola bekel mantul lompat panggil rekursi jatuh lantai mantul pantul mantul rekur.",
        "tugas": "- Rekursif bola mantul setengah tinggi awal.",
        "code": """
public class PantulanBolaBekelKaretLompatGravitasiPanggilLantaiHantamanFisikaGesekUdaraKetinggianMenurunSiklusElastisitasMantulBouncingBallFungsiLoopRekursiMurniKacaNyataKaret {

    public static void mantulkanBolaKaretKelantaiUbinKerasGravitasiFisikaLompat(int tinggiCmBolaDilayangkanLepasJariLemparLantaiJatuhGrafitasiTarikanBumiMassaBalikMumbul) {
        if (tinggiCmBolaDilayangkanLepasJariLemparLantaiJatuhGrafitasiTarikanBumiMassaBalikMumbul <= 0) {
            System.out.println("Nggelitik Berhenti Getar Pletok Pletok Pletok Nyelip Bawah Kolong Kasur Lemari Ilang Digondol Tikus Debu Lantai Bola Bekel Kusam Hitam Kecil Lenyap Berhenti Karet Gelinding Diam.");
            return;
        }
        System.out.println("Dung!!! Memantul Naik Udara Mumbul Menjangkau " + tinggiCmBolaDilayangkanLepasJariLemparLantaiJatuhGrafitasiTarikanBumiMassaBalikMumbul + " Cm");
        ... (tinggiCmBolaDilayangkanLepasJariLemparLantaiJatuhGrafitasiTarikanBumiMassaBalikMumbul / 2); // Loss fisika memantul sisa energinya setengah gravitasi kinetik
    }

    public static void main(String[] args) {
        System.out.println("Bocah Gendheng Melempar Keras Bola Bekel Karet Dari Atas Lemari Tinggi Jatuh Lurus Licin Keramik Lantai Keras:");
        mantulkanBolaKaretKelantaiUbinKerasGravitasiFisikaLompat(16);
    }
}"""
    },
    {
        "topic": "Fungsi Rekursif",
        "studi_kasus": "Pembayaran hutang rentenir rekursif bunga pinjol. Hutang bayar panggil parameter cicilan potong utang sisa panggil cicilan hutang debt collector panggil fungsi panggil tagihan lunas batas nol batas lunas fungsi tagih hutang uang sisa pinjaman rente panggil panggil minus tagihan lunas batas kurang debt kolektor rente uang peminjaman lunas utang.",
        "tugas": "- Parameter hutang kurang potongan lunas rekursi",
        "code": """
public class CicilanPinjamanIlegalPinjolTagihanHutangRentenirDebtCollectorMarah {

    public static void bayarAnsangCicilPinjolDiterorDebtKolektorTeleponKontakDaruratSodarPinjamanUtangMencekikBungaMeledakBesarBakar(int sisaHutangJebakanJahatRentenirRibaRatusanRibuJutaUang) {
        if (sisaHutangJebakanJahatRentenirRibaRatusanRibuJutaUang <= 0) {
            System.out.println("Alhamdulillah Amplop Gajian Kuras Habis Bebas Gali Lobang Tutup Lobang Hutang Lunas Terbayar Bersih Plong Pikiran Lega Bahagia Makan Nasi Garm Kosong.");
            return;
        }
        System.out.println("Membayar Transfer Ke Virtual Account M-Banking Tagihan Hutang Lintah Darat Pinjol Aplikasi Illegal Mencekik Leher. Sisa Pokok: Rp " + sisaHutangJebakanJahatRentenirRibaRatusanRibuJutaUang);
        ... (sisaHutangJebakanJahatRentenirRibaRatusanRibuJutaUang - 500000); // 500rb per cicilan telat denda panggil ansang bulanan narik
    }

    public static void main(String[] args) {
        System.out.println("Pegawai Pabrik Gemetar Keringatan Buka Aplikasi Cek Tunggakan Panik Sisa Hitungan Pembayaran Menggurita:");
        bayarAnsangCicilPinjolDiterorDebtKolektorTeleponKontakDaruratSodarPinjamanUtangMencekikBungaMeledakBesarBakar(2000000);
    }
}"""
    },
    {
        "topic": "Fungsi Rekursif",
        "studi_kasus": "Robot cat mengecat tembok gedung rekursi lapis oles warna cat kurang lapis poles oles poles rekursi kuas dinding kilap warna kilau tebal lapis poles basah ulang cat fungsi robot lengan mekanik kaleng rol poles batas dasar poles lapis parameter poles nol stop mengoles kuas dinding cat ulang tebal poles ulang batas oles batas nol.",
        "tugas": "- Panggil lapis kuas turun",
        "code": """
public class RobotKuasTembokCatGedungPelapisGoresanTebalWarnaKuasanBasahSilikonAntiAirDindingRekursiRolKuasanSelesaiKeringMengkilapOles {

    public static void polesLapisCatTembokKeringBelumUlangiNumpukLapisOlesGoresKuasRollKalengCairanBasaCat(int targetJumlahKetebalanOlesLapisCatGarisMeterPersegiTembokPolosPutihSemenKasarAplikasi) {
        if (targetJumlahKetebalanOlesLapisCatGarisMeterPersegiTembokPolosPutihSemenKasarAplikasi == 0) {
            System.out.println("Tembok Rata Super Tebal Mengkilat Basah Cat Warna Fresh Baru Tahan Air Hujan Deras Jamur Lumut Ganjel Kering Menjemur Matahari Biarkan Udara Selesai Proses Poles Pelukis Gedung Robot Mesin Pintar Istirahat Ngecas Batrai Habis Listrik Kerja Lembur Ngecat!");
            return;
        }
        System.out.println("Ssssrttttt.. Robot Lengan Ayun Roll Kuas Menggulung Tembok Kasar Poles Cat Acrilyc Kental Lapis Basah Ke-" + targetJumlahKetebalanOlesLapisCatGarisMeterPersegiTembokPolosPutihSemenKasarAplikasi);
        ... (targetJumlahKetebalanOlesLapisCatGarisMeterPersegiTembokPolosPutihSemenKasarAplikasi - 1);
    }

    public static void main(String[] args) {
        System.out.println("Mandor Proyek Menekan Remot Menghidupkan Robot Ngecat Gedung Tinggi Lantai 5 Tali Kerek Bergelantungan Kencang:");
        polesLapisCatTembokKeringBelumUlangiNumpukLapisOlesGoresKuasRollKalengCairanBasaCat(3);
    }
}"""
    }
]
