# Tahap 1: Topik 1-4 (40 Soal Java Bervariasi)

EXPANDED_JAVA_EXAMPLES_P1 = [
    # ================= TOPIC 1: Tipe Data, Variabel dan Operator =================
    {
        "topic": "Tipe Data, Variabel dan Operator",
        "studi_kasus": "Dalam game RPG, seorang Kesatria (Knight) memiliki Health Points (HP) dan Armor. Saat diserang monster, kerusakan (damage) yang diterima akan dikurangi oleh pertahanan armor. Kalkulasi sisa nyawa karakter setelah satu kali tebasan naga.",
        "tugas": "- Inisialisasi variabel hpAwal dengan 1000 dan nilaiArmor dengan 50\n- Deklarasikan variabel damageNaga sebesar 250\n- Hitung damageFinal = damageNaga - nilaiArmor\n- Kurangi hpAwal dengan damageFinal untuk mendapatkan sisaHp\n- Cetak sisaHp karakter ke layar",
        "code": """
public class GameRPG {
    public static void main(String[] args) {
        int hpAwal = ...;
        int nilaiArmor = 50;
        int damageNaga = ...;
        
        int damageFinal = damageNaga - ...;
        int sisaHp = hpAwal - ...;
        
        System.out.println("Sisa HP Kesatria: " + ...);
    }
}"""
    },
    {
        "topic": "Tipe Data, Variabel dan Operator",
        "studi_kasus": "Sistem sensor IoT di sebuah GreenHouse mengukur kelembapan tanah (humidity). Petani ingin mengonversi data sensor analog murni (0-1023) menjadi persentase (0.0% - 100.0%).",
        "tugas": "- Inisialisasi variabel sensorValue (analog) dengan 750\n- Buat variabel persentase bertipe double\n- Hitung persentase dengan rumus: (sensorValue / 1023.0) * 100\n- Tampilkan nilai analog dan nilai persentase kelembapan.",
        "code": """
public class SensorKelembapan {
    public static void main(String[] args) {
        int sensorValue = ...;
        double persentase;
        
        persentase = (sensorValue / 1023.0) * ...;
        
        System.out.println("Nilai Analog: " + sensorValue);
        System.out.println("Persentase Kelembapan: " + persentase + "%");
    }
}"""
    },
    {
        "topic": "Tipe Data, Variabel dan Operator",
        "studi_kasus": "Sistem perbankan syariah mencatat saldo tabungan nasabah. Setiap akhir bulan, nasabah disisihkan zakat mal sebesar 2.5% dari total saldo jika mencapai nisab.",
        "tugas": "- Inisialisasi variabel saldoTabungan dengan 85000000.0\n- Inisialisasi persentaseZakat sebesar 2.5\n- Hitung nominalZakat dengan mengalikan saldo dengan (persentaseZakat / 100)\n- Cetak nominal zakat yang harus dibayarkan",
        "code": """
public class KalkulatorZakat {
    public static void main(String[] args) {
        double saldoTabungan = ...;
        double persentaseZakat = ...;
        
        double nominalZakat = saldoTabungan * (... / 100.0);
        
        System.out.println("Total Zakat Mal: Rp" + nominalZakat);
    }
}"""
    },
    {
        "topic": "Tipe Data, Variabel dan Operator",
        "studi_kasus": "Rumah sakit menghitung dosis obat paracetamol cair berdasarkan berat badan anak. Resep medis menyebutkan 15 miligram per kilogram berat badan.",
        "tugas": "- Inisialisasi variabel beratBadanAnak dengan 22.5 (kg)\n- Inisialisasi dosisPerKg dengan 15\n- Hitung totalDosis = beratBadanAnak * dosisPerKg\n- Cetak total miligram obat yang dibutuhkan",
        "code": """
public class DosisObat {
    public static void main(String[] args) {
        double beratBadanAnak = ...;
        int dosisPerKg = ...;
        
        double totalDosis = ... * dosisPerKg;
        
        System.out.println("Dosis Paracetamol: " + totalDosis + " mg");
    }
}"""
    },
    {
        "topic": "Tipe Data, Variabel dan Operator",
        "studi_kasus": "Aplikasi e-commerce otomotif menghitung estimasi ongkos kirim ban mobil antar kota berdasarkan jarak logistik (km) dan tarif per kilometer.",
        "tugas": "- Deklarasikan jarakTempuh sebesar 125.5 (kilometer)\n- Deklarasikan tarifPerKm sebesar 4500\n- Hitung totalOngkir dengan mengalikan jarak dengan tarif\n- Tampilkan biaya pengiriman",
        "code": """
public class OngkirBan {
    public static void main(String[] args) {
        double jarakTempuh = ...;
        int tarifPerKm = ...;
        
        double totalOngkir = jarakTempuh * ...;
        
        System.out.println("Biaya Kirim Ban: Rp" + totalOngkir);
    }
}"""
    },
    {
        "topic": "Tipe Data, Variabel dan Operator",
        "studi_kasus": "Aplikasi kebugaran pintar (Smartwatch) menghitung jumlah kalori harian yang terbakar dengan mengalikan jumlah langkah kaki dengan rata-rata kalori per langkah.",
        "tugas": "- Inisialisasi totalLangkah dengan 8540\n- Deklarasikan kaloriPerLangkah sebesar 0.04 (kkal)\n- Hitung kaloriTerbakar = totalLangkah * kaloriPerLangkah\n- Tampilkan total kalori",
        "code": """
public class FitnessTrack {
    public static void main(String[] args) {
        int totalLangkah = ...;
        double kaloriPerLangkah = ...;
        
        double kaloriTerbakar = totalLangkah * ...;
        
        System.out.println("Kalori Terbakar: " + ... + " kkal");
    }
}"""
    },
    {
        "topic": "Tipe Data, Variabel dan Operator",
        "studi_kasus": "Pada server cloud AWS, tagihan bulanan dihitung dari pemakaian transfer data bandwidth keluar (Egress) dengan tarif $0.09 per Gigabyte.",
        "tugas": "- Deklarasikan totalBandwidth sebesar 450.5 (GB)\n- Deklarasikan tarifGB sebesar 0.09\n- Hitung tagihanServer\n- Cetak tagihan AWS bulanan ke layar",
        "code": """
public class TagihanCloud {
    public static void main(String[] args) {
        double totalBandwidth = ...;
        double tarifGB = ...;
        
        double tagihanServer = ... * tarifGB;
        
        System.out.println("Tagihan Egress Data: $" + ...);
    }
}"""
    },
    {
        "topic": "Tipe Data, Variabel dan Operator",
        "studi_kasus": "Seorang insinyur sipil membutuhkan program untuk menghitung volume beton yang dibutuhkan untuk sebuah pondasi kolom (Balok) berdasarkan panjang, lebar, dan tinggi.",
        "tugas": "- Inisialisasi panjang, lebar, dan tinggi (contoh: 4.5, 2.0, 1.5)\n- Hitung volumeBeton = panjang * lebar * tinggi\n- Cetak volume dalam satuan meter kubik",
        "code": """
public class VolumeBeton {
    public static void main(String[] args) {
        double panjang = 4.5;
        double lebar = ...;
        double tinggi = ...;
        
        double volumeBeton = panjang * lebar * ...;
        
        System.out.println("Kebutuhan Beton: " + volumeBeton + " m³");
    }
}"""
    },
    {
        "topic": "Tipe Data, Variabel dan Operator",
        "studi_kasus": "Aplikasi manajemen restoran otomatis menghitung uang kembalian pelanggan berdasarkan total nota pembayaran dan uang tunai yang diserahkan.",
        "tugas": "- Inisialisasi tagihanNota dengan 145000\n- Inisialisasi uangTunai dengan 200000\n- Hitung kembalian pelanggan dengan mengurangi tagihan terhadap uang tunai\n- Tampilkan nominal kembalian kasir",
        "code": """
public class KasirResto {
    public static void main(String[] args) {
        int tagihanNota = ...;
        int uangTunai = ...;
        
        int kembalian = uangTunai - ...;
        
        System.out.println("Kembalian Pelanggan: Rp" + kembalian);
    }
}"""
    },
    {
        "topic": "Tipe Data, Variabel dan Operator",
        "studi_kasus": "Agensi periklanan digital menghitung 'Click-Through Rate' (CTR) dari konversi jumlah klik iklan dibagi jumlah penayangan (impressions), dikalikan seratus.",
        "tugas": "- Deklarasikan jumlahKlik (contoh: 450) dan jumlahView (contoh: 15000)\n- Hitung metrik CtrPersen = (jumlahKlik / jumlahView) * 100\n- Cetak rasio CTR",
        "code": """
public class PerformaIklan {
    public static void main(String[] args) {
        double jumlahKlik = ...;
        double jumlahView = ...;
        
        double CtrPersen = (... / jumlahView) * 100;
        
        System.out.println("Click-Through Rate: " + CtrPersen + "%");
    }
}"""
    },

    # ================= TOPIC 2: Sintaks Pemilihan 1 (Basic Conditionals) =================
    {
        "topic": "Sintaks Pemilihan 1",
        "studi_kasus": "Sistem Anti-Cheat di Game Esports akan secara otomatis memblokir (banned) akun pemain jika terdeteksi menggunakan software ilegal (cheat mod).",
        "tugas": "- Inisialisasi boolean isCheatTerdeteksi = true\n- Gunakan IF statement untuk memeriksa kondisi tersebut\n- Jika benar, print 'AKUN DIBANNED 10 TAHUN'. Jika salah, print 'STATUS BERSIH'",
        "code": """
public class AntiCheat {
    public static void main(String[] args) {
        boolean isCheatTerdeteksi = ...;
        
        if (...) {
            System.out.println("AKUN DIBANNED 10 TAHUN");
        } else {
            System.out.println("STATUS BERSIH");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 1",
        "studi_kasus": "Aplikasi pengatur suhu cerdas (Smart Thermostat) akan secara otomatis menyalakan kompresor AC (Air Conditioner) jika suhu ruangan melebihi 26 derajat celsius.",
        "tugas": "- Inisialisasi suhuRuangan dengan 28\n- Buat kondisi IF dimana AC menyala jika suhu > 26\n- Berikan aksi else jika suhu stabil",
        "code": """
public class SmartThermostat {
    public static void main(String[] args) {
        int suhuRuangan = ...;
        
        if (suhuRuangan > ...) {
            System.out.println("Kompresor AC Menyala. Mendinginkan ruangan.");
        } else {
            System.out.println("Suhu Normal. AC mode Standby.");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 1",
        "studi_kasus": "Mesin kasir swalayan memastikan seorang pembeli mendapatkan promo gratis tas belanja kain jika total belanjanya di atas atau sama dengan Rp 250.000.",
        "tugas": "- Deklarasikan totalBelanja (misal 300000)\n- Pakai pengkondisian untuk mengecek kelayakan promo belanja\n- Tampilkan pesan dapat tas kain atau belum.",
        "code": """
public class KasirSwalayan {
    public static void main(String[] args) {
        int totalBelanja = ...;
        
        if (totalBelanja ... 250000) {
            System.out.println("Selamat! Anda berhak mendapat Tas Belanja Kain Gratis.");
        } else {
            System.out.println("Belanja Anda belum mencapai target promo Tas Kain.");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 1",
        "studi_kasus": "Sensor kelembapan tanah mengirim notifikasi peringatan kekeringan pada panel pompa air cerdas jika persentase soil moisture turun di bawah 15%.",
        "tugas": "- Inisialisasi kelembapanTanah (12%)\n- Implementasikan IF untuk memonitor ambang batas bahaya (< 15)\n- Cetak notifikasi siram air",
        "code": """
public class PompaAir {
    public static void main(String[] args) {
        int kelembapanTanah = ...;
        
        if (kelembapanTanah ... 15) {
            System.out.println("PERINGATAN KEKERINGAN! Pompa air diaktifkan.");
        } else {
            System.out.println("Tanah masih subur dan lembap.");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 1",
        "studi_kasus": "Aplikasi perbankan mendeteksi fraud, menolak transaksi kartu kredit jika nominal penarikan jauh lebih besar daripada batas limit sisa saldo harian.",
        "tugas": "- Tentukan limitHarian (5000000)\n- Tentukan nominalTarik (6500000)\n- Blokir dengan IF jika tarikan melebihi saldo limit yang ada",
        "code": """
public class FraudBank {
    public static void main(String[] args) {
        int limitHarian = ...;
        int nominalTarik = ...;
        
        if (nominalTarik > ...) {
            System.out.println("TRANSAKSI DITOLAK! Melebihi limit kartu.");
        } else {
            System.out.println("Transaksi Berhasil.");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 1",
        "studi_kasus": "Sistem registrasi event marathon internasional secara otomatis menolak pendaftaran jika indeks massa tubuh (BMI) kontestan terindikasi sangat kurang gizi (< 16.0).",
        "tugas": "- Inisialisasi bmiKontestan (15.5)\n- Tolak pendaftaran dengan IF kondisi bmi < 16.0\n- Terima untuk default ELSE",
        "code": """
public class RegistrasiMarathon {
    public static void main(String[] args) {
        double bmiKontestan = ...;
        
        if (bmiKontestan ... 16.0) {
            System.out.println("TIDAK LULUS MEDIS: Risiko malnutrisi tinggi.");
        } else {
            System.out.println("LOLOS TIM MEDIS: Silakan ambil nomor dada.");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 1",
        "studi_kasus": "Gudang e-commerce Robot membaca barcode kemasan. Jika stok barang di inventaris kurang dari 10 kardus, sistem menyalakan alarm merah untuk memanggil kurir restok.",
        "tugas": "- Set stokKardus (5)\n- Nyalakan alarm dengan IF apabila stok sisa kurang dari 10",
        "code": """
public class GudangRobot {
    public static void main(String[] args) {
        int stokKardus = ...;
        
        if (stokKardus ... 10) {
            System.out.println("ALARM MERAH: Stok barang menipis! Panggil logistik!");
        } else {
            System.out.println("Stok barang di gudang aman.");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 1",
        "studi_kasus": "Aktivitas penerbangan di observatorium bandara ditunda jika visibilitas (jarak pandang pilot) turun di bawah angka 500 meter karena ada badai salju.",
        "tugas": "- Hitung variabel jarakPandangMeter\n- Periksa validitas IF untuk menunda pesawat jika (< 500)",
        "code": """
public class RadarBandara {
    public static void main(String[] args) {
        int jarakPandangMeter = ...;
        
        if (jarakPandangMeter < ...) {
            System.out.println("DITUNDA: Visibilitas bahaya karena cuaca buruk.");
        } else {
            System.out.println("CLEAR TO TAKEOFF: Jarak pandang aman.");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 1",
        "studi_kasus": "Dalam sistem tilang elektronik (E-TLE) di jalan raya, mobil otomatis dikenakan denda pelanggaran jika kecepatan laju kamera mendeteksi angka melebihi 100 km/jam.",
        "tugas": "- Deklarasikan kecepatanMobil (115)\n- Kirim surat tilang melalui IF bila batas kecepatan (100) dilanggar",
        "code": """
public class TilangElektronik {
    public static void main(String[] args) {
        int kecepatanMobil = ...;
        
        if (kecepatanMobil > ...) {
            System.out.println("ETLE TERTIUP: Kecepatan over limit! Denda dikirim.");
        } else {
            System.out.println("Kecepatan kendaraan terpantau normal.");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 1",
        "studi_kasus": "Sistem irigasi pertanian menggunakan sensor Ph Tanah. Jika tingkat keasaman (Ph) air persawahan bernilai persis angka 7.0, air dinilai netral dan sangat aman.",
        "tugas": "- Set phTanah dengan 7.0\n- Lakukan evaluasi keseimbangan (equality ==) untuk Ph tanah Netral.",
        "code": """
public class IrigasiPertanian {
    public static void main(String[] args) {
        double phTanah = ...;
        
        if (phTanah ... 7.0) {
            System.out.println("Sempurna! Tingkat keasaman tanah Neutral (Aman).");
        } else {
            System.out.println("Tanah memiliki kadar senyawa asam basa yang bergeser.");
        }
    }
}"""
    },

    # ================= TOPIC 3: Sintaks Pemilihan 2 (Switch Case / If-Else-If Sequence) =================
    {
        "topic": "Sintaks Pemilihan 2",
        "studi_kasus": "Sistem lampu lalu lintas perempatan tol merespons warna sensor. Merah berarti 'BERHENTI TOTAL', Kuning 'SIAP-SIAP', Hijau 'JALAN TERUS'. Selain itu sensor rusak.",
        "tugas": "- Gunakan switch case untuk mengecek var lampu\n- Tulis instruksi setiap lampu dengan System.out\n- Jangan lupakan statement BREAK",
        "code": """
public class SensorLampuTol {
    public static void main(String[] args) {
        String warnaLampu = "KUNING";
        
        switch (warnaLampu) {
            case "...":
                System.out.println("BERHENTI TOTAL");
                ...; // Jangan lupa
            case "KUNING":
                System.out.println("SIAP-SIAP CEK REM");
                break;
            case "HIJAU":
                System.out.println("..."); // Isi instruksi jalan
                break;
            default:
                System.out.println("Sistem lampu konslet!");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 2",
        "studi_kasus": "Game survival membagi level kesulitan zombie horde ke dalam: 1 (Easy), 2 (Medium), 3 (Hardcore), dan 4 (Nightmare). Selain data itu layar menu gagal dimuat.",
        "tugas": "- Inisialisasi levelDifficulty\n- Switch nilai interger tersebut\n- Tetapkan jenis spawn monster zombie",
        "code": """
public class ZombieHorde {
    public static void main(String[] args) {
        int levelDifficulty = 3;
        
        switch (...) {
            case 1:
                System.out.println("Mode Easy: Zombie lambat.");
                break;
            case 2:
                System.out.println("Mode Medium: Zombie mulai berlari.");
                break;
            case ...:
                System.out.println("Mode Hardcore: Zombie punya armor beracun.");
                break;
            case 4:
                System.out.println("Mode Nightmare: Oksigen terbatas!");
                break;
            ...:
                System.out.println("Level ID tidak dikenali Engine.");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 2",
        "studi_kasus": "Dalam klasifikasi meteorologi cuaca satelit satelit BMKG, Skala Fujita Torando (F0 - F5) mengatur tingkat destruksi. Kita evaluasi kecepatan angin kategori.",
        "tugas": "- Gunakan Switch Case untuk char kodeTornado '0', '1', '2'\n- F0 kerusakan ringan, F1 atap terlepas, F2 badai parah",
        "code": """
public class SkalaFujita {
    public static void main(String[] args) {
        char kodeTornado = '2';
        
        switch (...) {
            case '0':
                System.out.println("F0: Ranting pohon patah, angin ringan.");
                break;
            case '1':
                System.out.println("F1: Atap rumah bergeser lepas.");
                ...;
            case '...':
                System.out.println("F2: Kerusakan parah, mobil bisa terangkat.");
                break;
            default:
                System.out.println("Kategori Badai Ekstrem (F3+) atau Invalid.");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 2",
        "studi_kasus": "Mesin pembuat kopi otomatis di bandara punya menu: E (Espresso), L (Latte), C (Cappuccino), M (Mocha). Beri feedback jika tombol lain ditekan.",
        "tugas": "- Evaluasi menggunakan statement pilihan\n- Pastikan huruf kapital sesuai",
        "code": """
public class VendingKopi {
    public static void main(String[] args) {
        char kodePilihanMenu = 'C';
        
        switch (kodePilihanMenu) {
            case '...':
                System.out.println("Menyeduh Espresso Pahit khas Italia...");
                break;
            case 'L':
                System.out.println("Mempersiapkan Latte Art Manis...");
                break;
            case 'C':
                System.out.println("...");
                ...;
            case 'M':
                System.out.println("Mocha Coklat diracik.");
                break;
            default:
                System.out.println("Tombol rusak atau Kopi Habis.");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 2",
        "studi_kasus": "Peringkat rank taktis prajurit militer: 10 (Jenderal), 8 (Kolonel), 5 (Kopral), 1 (Prajurit Baru). Beritahu tugas dinas kebangsaan berdasarkan input.",
        "tugas": "- Pake switch untuk mengecek pangkatPraurit\n- Lengkapi kode logika cabang",
        "code": """
public class PangkatMiliter {
    public static void main(String[] args) {
        int pangkatRank = 8;
        
        switch (pangkatRank) {
            case 10:
                System.out.println("Tugas: Mengatur Strategi Perang Udara (Jenderal)");
                break;
            case ...:
                System.out.println("Tugas: Memimpin Pasukan Resimen Inti (Kolonel)");
                break;
            case 5:
                System.out.println("Tugas: Jaga Garis Pertahanan Timur (Kopral)");
                ...;
            case 1:
                System.out.println("Tugas: Membersihkan Barak dan Latihan Beban.");
                break;
            default:
                System.out.println("Data identitas kesatuan terhapus intel.");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 2",
        "studi_kasus": "Deteksi tipe file virus berdasarkan ekstensi biner. EXE=Executable Malware, DLL=Dynamic Library Hijack, BAT=Batch Script Miner. Peringatkan user.",
        "tugas": "- Gunaian String ekstensiFile\n- Tampilkan ancaman cybersecurity komputer",
        "code": """
public class AntiVirusSistem {
    public static void main(String[] args) {
        String ekstensi = "BAT";
        
        switch (...) {
            case "EXE":
                System.out.println("Bahaya: Potensi Aplikasi Eksekutor Rootkit!");
                break;
            case "DLL":
                System.out.println("Bahaya: Injeksi Library Hijack!");
                break;
            case "...":
                System.out.println("Waspada: Auto-Script Penambang Crypto!");
                ...;
            default:
                System.out.println("File kemungkinan aman, ekstensi umum.");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 2",
        "studi_kasus": "Klasifikasi tingkat urgensi medis (Triage) di UGD Rumah Sakit. MERAH = Penanganan Darurat Operasi. KUNING = Segera Observasi. HIJAU = Rawat Jalan.",
        "tugas": "- Tentukan gelangTriage pasien ke MERAH/KUNING/HIJAU\n- Pilih ruangan dokter",
        "code": """
public class UGDHospital {
    public static void main(String[] args) {
        String gelangTriage = "MERAH";
        
        switch (gelangTriage) {
            case "...":
                System.out.println("KRITIS: Langsung bawa masuk ruang Resusitasi/Bilah Operasi!");
                break;
            case "KUNING":
                System.out.println("URGENT: Baringkan pasien di kasur observasi.");
                ...;
            case "HIJAU":
                System.out.println("NORMAL: Pasien dipersilakan ke loket antrean luar.");
                break;
            default:
                System.out.println("Tolong kalibrasi ulang alat medis pencetak gelang pasien.");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 2",
        "studi_kasus": "Aplikasi e-tilang tol membaca golongan kendaraan dari RFID E-Toll. 1: Sedan/Jeep. 2: Truk Gandeng 2 Poros. 3: Bus Pariwisata.",
        "tugas": "- Ganti cabang menggunakan switch angka golonganToll\n- Info biaya terpotong saldo",
        "code": """
public class GateEToll {
    public static void main(String[] args) {
        int golonganSensor = 2;
        
        switch (...) {
            case 1:
                System.out.println("Mobil / Jeep - Potong saldo Rp 15.500");
                break;
            case ...:
                System.out.println("Truk Gandeng Poros - Potong saldo Rp 45.000");
                break;
            case 3:
                System.out.println("Bus Besar Pariwisata - Potong saldo Rp 35.000");
                ...;
            ...:
                System.out.println("Kendaraan ilegal (Roda dua), Tilang Manual Aktif!");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 2",
        "studi_kasus": "Sistem pengaturan kamera Drone DSLR Profesional memiliki mod dial: 'M' (Manual), 'A' (Aperture Priority), 'S' (Speed/Shutter), 'P' (Program Auto).",
        "tugas": "- Konfigurasi parameter ISO dan fokus lensa bedasarkan mod fisik kamera.",
        "code": """
public class FlightKamera {
    public static void main(String[] args) {
        char dialKamera = 'A';
        
        switch (...) {
            case 'M':
                System.out.println("Mode Manual: Pilot mengatur Shutter & Aperture sendiri.");
                break;
            case '...':
                System.out.println("Lensa Bukaan Diatur, kecepatan tertahan!");
                break;
            case 'S':
                System.out.println("Mode Blur cepat, lensa membeku untuk olahraga.");
                ...;
            case 'P':
                System.out.println("Sistem Drone Otomatis memandu foto HDR.");
                break;
            default:
                System.out.println("Papan Dial kamera drone rusak udara!");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan 2",
        "studi_kasus": "Panel lift roket ulang alik angkasa NASA mengunci pintu secara otomatis dengan tahapan phase. P1: Mesin Pemanas, P2: Main Engine Start, P3: Solid Rocket Booster Ignition.",
        "tugas": "- Eval string phase\n- Kontrol ledakan hidroken",
        "code": """
public class RoketNASA {
    public static void main(String[] args) {
        String tahapanPhase = "P2";
        
        switch (tahapanPhase) {
            case "P1":
                System.out.println("Pre-flight: Pembilasan Hidrogen cair dingin.");
                break;
            case "...":
                System.out.println("Main Engine Start! Nyala api utama bawah keluar.");
                ...;
            case "P3":
                System.out.println("Solid Boost Ignition! Terbang bebas G-Force!");
                break;
            default:
                System.out.println("Abort Komando! Tahan misi penerbangan angkasa.");
        }
    }
}"""
    },

    # ================= TOPIC 4: Sintaks Pemilihan Bersarang (Nested Conditionals) =================
    {
        "topic": "Sintaks Pemilihan Bersarang",
        "studi_kasus": "Algoritma radar kapal selam untuk menghindari tabrakan tebing karang. Pertama periksa jarak (Sonar). Jika jarak dekat (< 50 meter), baru lakukan pengecekan ke arah sudut kemudi apakah lurus atau berbelok.",
        "tugas": "- Buat kondisi IF induk untuk mendeteksi jarak sonar kritis\n- Di dalam blok tersebut, buat IF bersarang untuk mendeteksi jika kapal jalan lurus lalu belokkan otomatis.",
        "code": """
public class KapalSelamRadar {
    public static void main(String[] args) {
        int jarakSonarM = 40;
        boolean arahLurus = true;
        
        if (jarakSonarM < ...) {
            System.out.println("PERINGATAN! Objek karang sangat dekat.");
            if (...) {
                System.out.println("Tindakan Autopilot: Banting setir tajam ke kiri 90 derajat!");
            } else {
                System.out.println("Tindakan: Kapal sedang menghindar secara melengkung, perlambat baling-baling.");
            }
        } else {
            System.out.println("Kawasan perairan hijau, zona lurus laut dalam.");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan Bersarang",
        "studi_kasus": "Gateway persetujuan pinjaman Bank Konvensional Syariah. Pelamar harus lulus BI Checking (True) dulu. Jika lolos baru dicek Gaji Bulanannya (> 10 juta = ACC Kredit Rumah HGB).",
        "tugas": "- Periksa flag boolean lulusBICheck\n- Evaluasi sub-kondisi pendapatan gaji (incomeRp) jika BIChecking valid",
        "code": """
public class LoanBank {
    public static void main(String[] args) {
        boolean lulusBICheck = true;
        int pendapatanGaji = 15000000;
        
        if (...) {
            System.out.println("Nasabah bebas dari Blacklist catatan hutang historis.");
            if (pendapatanGaji > ...) {
                System.out.println("Status: ACC! Layak Ajukan KPR Hunian Besar.");
            } else {
                System.out.println("Status: Batas Kredit Maksimal Kelas Subsidi Motor.");
            }
        } else {
            System.out.println("Status: DITOLAK. Selesaikan denda pinjol / tunggakan bank.");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan Bersarang",
        "studi_kasus": "Keamanan gerbang pangkalan nuklir (Area 51). Pertama sistem memindai wajah biometrik (Dikenali). Jika ya, baru sistem meminta deteksi radiasi hazmat pakaian keamanan.",
        "tugas": "- Validasi Boolean faceIDRecognized\n- Loop if cek radiasi setelan hazmatLevel keamanan.",
        "code": """
public class AreaMiliterNuklir {
    public static void main(String[] args) {
        boolean wajahDikenali = true;
        int hazmatRadiasiLevel = 25;
        
        if (...) {
            System.out.println("Intel Polisi Wajah Valid. Personil Terautentikasi.");
            if (hazmatRadiasiLevel < ...) {
                System.out.println("Sinar radiasi wajar. Akses Lift Bawah Tanah Diberikan.");
            } else { // Jika hazmat radiasi melebihi angka 30 misalnya
                System.out.println("Pakaian Terpapar Kebocoran! Karantina Sterilisasi Ozon.");
            }
        } else {
            System.out.println("Tembak Di Tempat: Penyusup Pangkalan!");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan Bersarang",
        "studi_kasus": "Dalam seleksi masuk tim E-Sports divisi MLBB Profesional. Calon pro player dicek win-rate kemenangannya (Harus > 75%). Setelahnya, dicek apakah dia spesialis role Jungler atau Support.",
        "tugas": "- Cek float rasio winRate tinggi\n- Evaluasi string jenis Role hero spesialis untuk kontrak bayaran.",
        "code": """
public class SeleksiEsports {
    public static void main(String[] args) {
        double winRateMatch = 82.5;
        String roleSpesialis = "JUNGLER";
        
        if (winRateMatch > ...) {
            System.out.println("Mekanik luar biasa! Cocok masuk draft profesional.");
            if (roleSpesialis.equals("...")) {
                System.out.println("Ditugaskan sebagai Assassin / Retribution Utama.");
            } else {
                System.out.println("Ditugaskan sebagai Cover / Tank Roamer strategi.");
            }
        } else {
            System.out.println("Maaf tingkat Win Rate terlalu publik/casual.");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan Bersarang",
        "studi_kasus": "Server cloud computing melakukan monitoring panas GPU server node. Jika suhu > 85 Celcius, baru ia akan mengecek Load CPU. Jika Load CPU juga 100%, sistem harus direstart pendingin paksa.",
        "tugas": "- Cek suhu server threshold.\n- Nested cek persentase beban core prosesor bottleneck.",
        "code": """
public class AutoCoolingServer {
    public static void main(String[] args) {
        int suhuGpuNodeInfo = 88;
        int loadCpuServerInfo = 100;
        
        if (suhuGpuNodeInfo > ...) {
            System.out.println("WARNING SERVER OVERHEAT RACK A1.");
            if (...) {
                System.out.println("FATAL: CPU LOAD 100% BLOK KIPAS! Inject Cairan Coolant!");
            } else {
                System.out.println("CPU Stabil, Perlambat Clock GPU saja.");
            }
        } else {
            System.out.println("Sirkulasi Udara Datacenter Dingin Aman.");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan Bersarang",
        "studi_kasus": "Pabrik roti memeriksa kelembapan gandum fermentasi ragi otomatis. Pertama pastikan berat ragi 50 gram, jika terpenuhi, ukur berapa jam fermentasinya. Kurang dari 2 jam = bantat lunak.",
        "tugas": "- Periksa gram tepung\n- Buat cabang umur jam fementasi ragi",
        "code": """
public class MesinPabrikRoti {
    public static void main(String[] args) {
        int beratRagiGram = 50;
        int lamaJamFermentasi = 3;
        
        if (beratRagiGram == ...) {
            System.out.println("Timbangan Takaran Pas sesuai resep ISO.");
            if (lamaJamFermentasi >= ...) { // Harus 3 jam atas
                System.out.println("Adonan Mengembang Sempurna. Masukkan ke Over Panggang.");
            } else {
                System.out.println("Ragi mati/belum aktif. Tepung bantat!");
            }
        } else {
            System.out.println("Timbang Ulang Mixer Bahan!");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan Bersarang",
        "studi_kasus": "Kendali otomatis traktor otonom (Self-Driving Tractor) mendeteksi halangan di sawah. Jika mendeteksi Objek Depan = True, selanjutnya sistem melihat tipe objek: Hewan/Batu atau Manusia.",
        "tugas": "- If awal true objekDepan\n- if bersarang tipeObjek (String)",
        "code": """
public class AI_TraktorSawah {
    public static void main(String[] args) {
        boolean halanganDepan = true;
        String dimensiSensor = "MANUSIA";
        
        if (...) {
            System.out.println("Radar Lidar menabrak gelombang pantul Objek!");
            if (dimensiSensor.equals("...")) {
                System.out.println("Prioritas nyawa tinggi! Matikan Pisau Pembajak Secara Instan.");
            } else {
                System.out.println("Itu hanya genangan tanah liar. Belok pelan hindari batu.");
            }
        } else {
            System.out.println("Bajak sawah rute GPS dilanjutkan.");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan Bersarang",
        "studi_kasus": "Aplikasi E-Wallet OVO/DANA/GOPAY memverifikasi pinjaman saldo darurat paylater. Pertama cek tipe akun Premium (True). Di dalamnya, cek riwayat keterlambatan telat bayar harus False.",
        "tugas": "- IF luar validasiAkunKYC (bool)\n- IF dalam validasi status Blacklist Cicilan (bool)",
        "code": """
public class LogikaEWalletPaylater {
    public static void main(String[] args) {
        boolean isAkunVerifiedNasional = true;
        boolean adaTunggakanCicilanLalu = false;
        
        if (...) {
            System.out.println("User telah selfie KTP dan Tanda Tangan Digital Resmi.");
            if (!...) { // Tidak boleh ada tunggakan
                System.out.println("Akses Dana Tunai Cair Maksimal Sebesar Limit!");
            } else {
                System.out.println("Paylater Dibekukan Sementara karena Sering Melanggar Janji Bayar.");
            }
        } else {
            System.out.println("Tolong Upgrade Akun Dasar ke Premium / Bisnis.");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan Bersarang",
        "studi_kasus": "Kabin kereta kecepatan tinggi (Shinkansen) sensor Gempa Bumi (Early Warning System). Deteksi gempa primer True, baru kemudian analisis kekuatan magnitudo (> 6 SR = Rem Darurat Hentakan Blok).",
        "tugas": "- if gempaDetected\n- nested kecepatan skalaRichter untuk aktuator REM baja",
        "code": """
public class KeretaCepatGempa {
    public static void main(String[] args) {
        boolean seismikGetarAktif = true;
        double skalaMagnitudo = 7.1;
        
        if (...) {
            System.out.println("Gelombang P (Primer) Bumi Terdeteksi Di Sekitar Rute Rel.");
            if (skalaMagnitudo >= ...) {
                System.out.println("TSUNAMI/PATAHAN BESAR! Hentakan Rem Darurat Hidrolik Maksimal Aktif!");
            } else {
                System.out.println("Gempa Lokal Ringan, Hanya Bunyikan Sirine Peringatan Stabilizer.");
            }
        } else {
            System.out.println("Fase Perjalanan Lintas Negara Mulus.");
        }
    }
}"""
    },
    {
        "topic": "Sintaks Pemilihan Bersarang",
        "studi_kasus": "Kamera CCTV Tilang Face Recognition di jalanan mendeteksi Identitas Residivis. Cek wajah valid daftar buron True, pasca itu periksa level kejahatan (Ringan/Berat).",
        "tugas": "- if tersangkaKetemu divalidasi ke data server kapolri\n- if dalam mengevaluasi string levelPidana",
        "code": """
public class CCTVFaceTrack {
    public static void main(String[] args) {
        boolean wajahDikenaliInterpol = true;
        String levelKejahatan = "BERAT_TERORIS";
        
        if (...) {
            System.out.println("Kunci Target Visual! Lampu Merah Diblokir CCTV!");
            if (levelKejahatan.equals("...")) {
                System.out.println("Panggil Helikopter Komando SWAT, Subjek Bersenjata Bahaya.");
            } else {
                System.out.println("Kirim Mobil Patroli Sabhara Biasa untuk Tilang/Jemput Pemalsuan.");
            }
        } else {
            System.out.println("Kendaraan warga sipil melintas normal di tol kota.");
        }
    }
}"""
    }
]
